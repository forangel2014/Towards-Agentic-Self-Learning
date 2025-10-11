import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import logging
import psutil
import torch.cuda
from datetime import datetime

# 配置日志记录
def setup_logger():
    log_dir = "./retriever_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"retrieval_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

import faiss
import torch
os.environ["TRANSFORMERS_IGNORE_IMAGE_UTILS"] = "1"

import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def load_corpus(corpus_path: str):
    print(f"Loading corpus from {corpus_path}")
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    print(f"Corpus loaded from {corpus_path}")
    return corpus

def read_jsonl(file_path):
    print(f"Reading JSONL file from {file_path}")
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"JSONL file read from {file_path}")
    return data

def load_docs(corpus, doc_idxs):
    print(f"Loading documents from corpus")
    results = [corpus[int(idx)] for idx in doc_idxs]
    print(f"Documents loaded from corpus")
    return results

def load_model(model_path: str, use_fp16: bool = False):
    print(f"Loading model from {model_path}")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    print(f"Model loaded from {model_path}")
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print(f"Initializing BM25Retriever")
        from pyserini.search.lucene import LuceneSearcher
        print(f"Loading LuceneSearcher from {self.index_path}")
        self.searcher = LuceneSearcher(self.index_path)
        print(f"LuceneSearcher loaded from {self.index_path}")
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            print(f"Loading corpus from {self.corpus_path}")
            self.corpus = load_corpus(self.corpus_path)
            print(f"Corpus loaded from {self.corpus_path}")
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        logger.info(f"初始化 DenseRetriever，检查系统资源状态")
        
        # 记录系统内存状态
        memory = psutil.virtual_memory()
        logger.info(f"系统内存使用情况: 总计={memory.total/1024**3:.2f}GB, 已用={memory.used/1024**3:.2f}GB, 可用={memory.available/1024**3:.2f}GB")
        
        # 记录GPU状态
        if torch.cuda.is_available():
            logger.info(f"GPU可用，数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"GPU {i} 显存: {torch.cuda.get_device_properties(i).total_memory/1024**2:.2f}MB")
        else:
            logger.warning("GPU不可用，将使用CPU模式")

        try:
            logger.info(f"正在加载FAISS索引: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            if config.faiss_gpu:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            logger.info(f"FAISS索引加载成功，维度: {self.index.d}, 总数: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"FAISS索引加载失败: {str(e)}")
            raise

        try:
            logger.info(f"正在加载语料库: {self.corpus_path}")
            self.corpus = load_corpus(self.corpus_path)
            logger.info(f"语料库加载成功，大小: {len(self.corpus)}")
        except Exception as e:
            logger.error(f"语料库加载失败: {str(e)}")
            raise

        try:
            logger.info(f"正在初始化编码器，模型: {config.retrieval_model_path}")
            self.encoder = Encoder(
                model_name = self.retrieval_method,
                model_path = config.retrieval_model_path,
                pooling_method = config.retrieval_pooling_method,
                max_length = config.retrieval_query_max_length,
                use_fp16 = config.retrieval_use_fp16
            )
            logger.info("编码器初始化成功")
        except Exception as e:
            logger.error(f"编码器初始化失败: {str(e)}")
            raise

        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
            
        try:
            # logger.info(f"处理单个查询: {query[:100]}...")
            # logger.info(f"当前GPU显存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            # 检查查询格式
            if not isinstance(query, str):
                raise ValueError(f"查询必须是字符串类型，当前类型: {type(query)}")
            
            query_emb = self.encoder.encode(query)
            #logger.info(f"查询编码成功，维度: {query_emb.shape}")
            
            scores, idxs = self.index.search(query_emb, k=num)
            #logger.info(f"检索成功，返回结果数: {len(idxs[0])}")
            
            idxs = idxs[0]
            scores = scores[0]
            
            results = load_docs(self.corpus, idxs)
            #logger.info("文档加载成功")
            
            if return_score:
                return results, scores.tolist()
            else:
                return results
                
        except Exception as e:
            logger.error(f"检索过程发生错误: {str(e)}", exc_info=True)
            raise

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        try:
            if isinstance(query_list, str):
                query_list = [query_list]
                
            # logger.info(f"处理批量查询，查询数量: {len(query_list)}")
            # logger.info(f"第一个查询示例: {query_list[0][:100]}...")
            
            if num is None:
                num = self.topk
            
            results = []
            scores = []
            
            for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
                try:
                    # 记录每个批次开始时的显存状态
                    # if torch.cuda.is_available():
                    #     logger.info(f"批次 {start_idx} 开始，当前GPU显存: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                    
                    query_batch = query_list[start_idx:start_idx + self.batch_size]
                    #logger.info(f"处理批次 {start_idx}, 大小: {len(query_batch)}")
                    
                    batch_emb = self.encoder.encode(query_batch)
                    #logger.info(f"批次编码成功，维度: {batch_emb.shape}")
                    
                    batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
                    #logger.info(f"批次检索成功，结果维度: {batch_scores.shape}")
                    
                    batch_scores = batch_scores.tolist()
                    batch_idxs = batch_idxs.tolist()
                    
                    flat_idxs = sum(batch_idxs, [])
                    batch_results = load_docs(self.corpus, flat_idxs)
                    batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
                    
                    results.extend(batch_results)
                    scores.extend(batch_scores)
                    
                    del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"批次 {start_idx} 处理失败: {str(e)}", exc_info=True)
                    raise
                    
            #logger.info(f"批量查询处理完成，总结果数: {len(results)}")
            
            if return_score:
                return results, scores
            else:
                return results
                
        except Exception as e:
            logger.error(f"批量检索过程发生错误: {str(e)}", exc_info=True)
            raise

def get_retriever(config):
    print(f"Getting retriever for {config.retrieval_method}")
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    try:

        # 记录请求信息
        #logger.info(f"收到检索请求: 查询数量={len(request.queries)}, topk={request.topk}, return_scores={request.return_scores}")
        
        # 验证输入
        if not request.queries:
            logger.error("请求中没有查询")
            raise ValueError("请求中必须包含至少一个查询")
            
        if any(not isinstance(q, str) for q in request.queries):
            logger.error("查询列表中包含非字符串类型")
            raise ValueError("所有查询必须是字符串类型")

        # 设置topk
        if not request.topk:
            request.topk = config.retrieval_topk
            #logger.info(f"使用默认topk值: {request.topk}")

        # 记录系统状态
        # if torch.cuda.is_available():
        #     logger.info(f"当前GPU显存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        # memory = psutil.virtual_memory()
        # logger.info(f"当前系统内存使用: {memory.percent}%")

        # 执行检索
        results, scores = retriever.batch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=request.return_scores
        )
        
        # 格式化响应
        resp = []
        for i, single_result in enumerate(results):
            if request.return_scores:
                combined = []
                for doc, score in zip(single_result, scores[i]):
                    combined.append({"document": doc, "score": score})
                resp.append(combined)
            else:
                resp.append(single_result)

        #logger.info(f"检索成功完成，返回结果数: {len(resp)}")
        return {"result": resp, "error": False}

    except Exception as e:
        logger.error(f"检索过程发生错误: {str(e)}", exc_info=True)
        return {
            "result": str(e),
            "error": True,
        }


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    print(f"Building retriever")
    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    print(f"Launching server on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
