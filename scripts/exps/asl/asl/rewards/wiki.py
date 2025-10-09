# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import gzip
import json
import os
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm


@dataclass
class WikiTriple:
    """维基百科三元组数据结构."""

    subject: str  # 主语
    predicate: str  # 谓语/关系
    object: str  # 宾语
    confidence: Optional[float] = None  # 置信度
    source: Optional[str] = None  # 来源

    def __repr__(self):
        return f"WikiTriple(subject='{self.subject}', predicate='{self.predicate}', object='{self.object}')"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式."""
        result = {"subject": self.subject, "predicate": self.predicate, "object": self.object}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.source is not None:
            result["source"] = self.source
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WikiTriple":
        """从字典创建三元组."""
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data.get("confidence"),
            source=data.get("source"),
        )


def download_wiki_triples_data(target_dir: str = "/diancpfs/user/sunwangtao/data/searchr1") -> str:
    """下载维基百科三元组数据到指定目录.

    Args:
        target_dir: 目标目录路径

    Returns:
        下载的数据文件路径
    """
    # 创建目标目录
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # 三元组数据源配置
    data_sources = {
        "wikidata_triples": {
            "url": "https://huggingface.co/datasets/wikidata-triples/resolve/main/wikidata-triples.jsonl.gz",
            "filename": "wikidata-triples.jsonl.gz",
            "description": "Wikidata三元组数据",
        },
        "dbpedia_triples": {
            "url": "https://huggingface.co/datasets/dbpedia-triples/resolve/main/dbpedia-triples.jsonl.gz",
            "filename": "dbpedia-triples.jsonl.gz",
            "description": "DBpedia三元组数据",
        },
    }

    # 尝试下载数据，如果失败则生成示例数据
    downloaded_files = []

    for source_name, source_info in data_sources.items():
        file_path = target_path / source_info["filename"]

        if file_path.exists():
            print(f"文件已存在: {file_path}")
            downloaded_files.append(str(file_path))
            continue

        print(f"正在下载 {source_info['description']}...")
        try:
            # 尝试从Hugging Face下载
            download_url = source_info["url"]
            urllib.request.urlretrieve(download_url, file_path)
            print(f"成功下载: {file_path}")
            downloaded_files.append(str(file_path))
        except Exception as e:
            print(f"下载失败: {e}")
            continue

    # 如果没有成功下载任何文件，生成示例三元组数据
    if not downloaded_files:
        print("无法下载在线数据，正在生成示例三元组数据...")
        sample_triples = generate_sample_triples()
        sample_file_path = target_path / "sample_triples.jsonl"

        with open(sample_file_path, "w", encoding="utf-8") as f:
            for triple in sample_triples:
                f.write(json.dumps(triple.to_dict(), ensure_ascii=False) + "\n")

        print(f"已生成示例数据: {sample_file_path}")
        downloaded_files.append(str(sample_file_path))

    return downloaded_files[0] if downloaded_files else None


def generate_sample_triples() -> list[WikiTriple]:
    """生成示例三元组数据.

    Returns:
        示例三元组列表
    """
    sample_triples = [
        WikiTriple("北京", "是", "中国的首都", 1.0, "示例数据"),
        WikiTriple("上海", "位于", "中国东部", 1.0, "示例数据"),
        WikiTriple("人工智能", "属于", "计算机科学", 1.0, "示例数据"),
        WikiTriple("Python", "是", "编程语言", 1.0, "示例数据"),
        WikiTriple("机器学习", "是", "人工智能的分支", 1.0, "示例数据"),
        WikiTriple("深度学习", "基于", "神经网络", 1.0, "示例数据"),
        WikiTriple("自然语言处理", "处理", "人类语言", 1.0, "示例数据"),
        WikiTriple("计算机视觉", "研究", "图像和视频", 1.0, "示例数据"),
        WikiTriple("数据挖掘", "从", "大数据中提取知识", 1.0, "示例数据"),
        WikiTriple("算法", "是", "解决问题的步骤", 1.0, "示例数据"),
        WikiTriple("数据库", "存储", "结构化数据", 1.0, "示例数据"),
        WikiTriple("网络", "连接", "计算机设备", 1.0, "示例数据"),
        WikiTriple("云计算", "提供", "按需计算资源", 1.0, "示例数据"),
        WikiTriple("区块链", "是", "分布式账本技术", 1.0, "示例数据"),
        WikiTriple("物联网", "连接", "物理设备", 1.0, "示例数据"),
        WikiTriple("大数据", "包含", "海量数据", 1.0, "示例数据"),
        WikiTriple("虚拟现实", "创造", "沉浸式体验", 1.0, "示例数据"),
        WikiTriple("增强现实", "叠加", "数字信息到现实世界", 1.0, "示例数据"),
        WikiTriple("量子计算", "使用", "量子力学原理", 1.0, "示例数据"),
        WikiTriple("边缘计算", "在", "网络边缘处理数据", 1.0, "示例数据"),
    ]
    return sample_triples


class WikiDataSampler:
    """维基数据采样器."""

    def __init__(self, wiki_data_path: Optional[str] = None, seed: Optional[int] = None):
        """初始化采样器.

        Args:
            wiki_data_path: 维基数据文件路径，如果为None则使用默认路径
            seed: 随机种子
        """
        self.wiki_data_path = wiki_data_path
        self.wiki_data = []
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # 如果提供了路径，则加载数据
        if wiki_data_path and os.path.exists(wiki_data_path):
            self.load_wiki_data(wiki_data_path)

    def load_wiki_data(self, data_path: str) -> None:
        """加载维基数据.

        Args:
            data_path: 数据文件路径
        """
        self.wiki_data = []

        # 根据文件扩展名决定加载方式
        if data_path.endswith(".json"):
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.wiki_data = data
                else:
                    # 如果是字典格式，尝试提取三元组
                    self.wiki_data = self._extract_triples_from_dict(data)

        elif data_path.endswith(".jsonl"):
            with open(data_path, encoding="utf-8") as f:
                for line in tqdm(f, desc="加载三元组数据"):
                    line = line.strip()
                    if line:
                        try:
                            triple_data = json.loads(line)
                            self.wiki_data.append(triple_data)
                        except json.JSONDecodeError:
                            continue

        elif data_path.endswith(".jsonl.gz"):
            with gzip.open(data_path, "rt", encoding="utf-8") as f:
                for line in tqdm(f, desc="加载压缩的三元组数据"):
                    line = line.strip()
                    if line:
                        try:
                            triple_data = json.loads(line)
                            self.wiki_data.append(triple_data)
                        except json.JSONDecodeError:
                            continue

        elif data_path.endswith(".txt"):
            with open(data_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "\t" in line:
                        # 假设是制表符分隔的格式：subject\tpredicate\tobject
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            triple = {
                                "subject": parts[0].strip(),
                                "predicate": parts[1].strip(),
                                "object": parts[2].strip(),
                            }
                            self.wiki_data.append(triple)

        print(f"已加载 {len(self.wiki_data)} 个三元组")

    def _extract_triples_from_dict(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """从字典中提取三元组.

        Args:
            data: 包含三元组的字典数据

        Returns:
            三元组列表
        """
        triples = []

        # 常见的三元组键名
        possible_keys = ["triples", "facts", "knowledge", "data", "entries"]

        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                triples.extend(data[key])
                break

        # 如果没有找到预定义的键，尝试直接解析
        if not triples and isinstance(data, dict):
            # 检查是否是单个三元组
            if all(k in data for k in ["subject", "predicate", "object"]):
                triples.append(data)

        return triples

    def sample_triple(self) -> Optional[WikiTriple]:
        """随机采样一个三元组.

        Returns:
            随机采样的三元组，如果没有数据则返回None
        """
        if not self.wiki_data:
            print("警告：没有加载维基数据，请先调用 load_wiki_data() 方法")
            return None

        # 随机选择一个三元组
        triple_data = random.choice(self.wiki_data)

        # 转换为WikiTriple对象
        if isinstance(triple_data, dict):
            return WikiTriple.from_dict(triple_data)
        elif isinstance(triple_data, list | tuple) and len(triple_data) >= 3:
            return WikiTriple(subject=str(triple_data[0]), predicate=str(triple_data[1]), object=str(triple_data[2]))
        else:
            print(f"警告：无法解析三元组数据格式：{triple_data}")
            return None

    def sample_triples(self, n: int = 1) -> list[WikiTriple]:
        """随机采样多个三元组.

        Args:
            n: 要采样的三元组数量

        Returns:
            三元组列表
        """
        if not self.wiki_data:
            print("警告：没有加载维基数据，请先调用 load_wiki_data() 方法")
            return []

        # 确保n不超过数据总量
        n = min(n, len(self.wiki_data))

        # 随机采样n个三元组
        sampled_data = random.sample(self.wiki_data, n)

        triples = []
        for triple_data in sampled_data:
            if isinstance(triple_data, dict):
                triples.append(WikiTriple.from_dict(triple_data))
            elif isinstance(triple_data, list | tuple) and len(triple_data) >= 3:
                triples.append(
                    WikiTriple(subject=str(triple_data[0]), predicate=str(triple_data[1]), object=str(triple_data[2]))
                )

        return triples

    def get_triple_count(self) -> int:
        """获取三元组总数.

        Returns:
            三元组总数
        """
        return len(self.wiki_data)


def sample_wiki_triple(
    wiki_data: str | list[dict[str, Any]] | list[WikiTriple], seed: Optional[int] = None
) -> Optional[WikiTriple]:
    """从wiki_data中随机采样一个三元组的便捷函数.

    Args:
        wiki_data: 维基数据，可以是文件路径、字典列表或WikiTriple列表
        seed: 随机种子

    Returns:
        随机采样的三元组，如果没有数据则返回None
    """
    if isinstance(wiki_data, str):
        # 如果是文件路径
        sampler = WikiDataSampler(wiki_data_path=wiki_data, seed=seed)
        return sampler.sample_triple()

    elif isinstance(wiki_data, list):
        # 如果是列表
        if not wiki_data:
            return None

        if seed is not None:
            random.seed(seed)

        triple_data = random.choice(wiki_data)

        if isinstance(triple_data, WikiTriple):
            return triple_data
        elif isinstance(triple_data, dict):
            return WikiTriple.from_dict(triple_data)
        elif isinstance(triple_data, list | tuple) and len(triple_data) >= 3:
            return WikiTriple(subject=str(triple_data[0]), predicate=str(triple_data[1]), object=str(triple_data[2]))
        else:
            print(f"警告：无法解析三元组数据格式：{triple_data}")
            return None

    else:
        print(f"警告：不支持的数据类型：{type(wiki_data)}")
        return None


def download_and_sample_triples(
    target_dir: str = "/diancpfs/user/sunwangtao/data/searchr1", seed: Optional[int] = None
) -> Optional[WikiTriple]:
    """下载三元组数据并随机采样一个三元组.

    Args:
        target_dir: 目标目录路径
        seed: 随机种子

    Returns:
        随机采样的三元组
    """
    print(f"开始下载三元组数据到: {target_dir}")

    # 下载数据
    data_file = download_wiki_triples_data(target_dir)

    if not data_file:
        print("下载失败，使用示例数据")
        # 生成示例数据
        sample_triples = generate_sample_triples()
        if sample_triples:
            if seed is not None:
                random.seed(seed)
            return random.choice(sample_triples)
        return None

    print(f"数据已下载到: {data_file}")

    # 加载并采样数据
    sampler = WikiDataSampler(wiki_data_path=data_file, seed=seed)
    triple = sampler.sample_triple()

    if triple:
        print(f"随机采样的三元组: {triple}")
    else:
        print("采样失败")

    return triple


# 示例用法
if __name__ == "__main__":
    # 下载数据并随机采样
    triple = download_and_sample_triples(seed=42)
    if triple:
        print(f"采样的三元组: {triple}")

    # 示例2：直接使用便捷函数
    # sample_data = [
    #     {"subject": "北京", "predicate": "是", "object": "中国的首都"},
    #     {"subject": "上海", "predicate": "位于", "object": "中国东部"},
    #     {"subject": "人工智能", "predicate": "属于", "object": "计算机科学"}
    # ]
    # triple = sample_wiki_triple(sample_data)
    # print(triple)

    pass
