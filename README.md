## Towards Agentic Self-Learning

### Data Processing
```bash
cd data/searchr1
bash download.sh
cd ../nq_hotpotqa_train
python nq_search.py
python format_data.py
cd ../meta
python generate.py
cd ../..
```

### Launch Retriever Server (8 GPUs)
```bash
cd tool
conda create -n retriever python=3.10
conda activate retriever
pip install -r requirements.txt
bash retrieval_launch.sh
```

### Training with ASL (32 GPUs)
```bash
bash scripts/exps/asl/run_asl_online_final.sh
```