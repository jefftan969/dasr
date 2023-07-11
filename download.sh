# Dataset (18G download)
wget -O data_processed.tar.gz "https://www.dropbox.com/scl/fi/kay6zlplkk484zsvxf7vd/data_processed.tar.gz?rlkey=l4o29auxzpvgzkj24t8f1hd0a&dl=0"
tar -xzvf data_processed.tar.gz
rm data_processed.tar.gz

# Checkpoints (788M download)
wget -O eval_checkpoints.tar.gz "https://www.dropbox.com/scl/fi/x61jpkz2v7fcg396i6lb9/eval_checkpoints.tar.gz?rlkey=6fomkdxrobnjxcou5hyr12t6c&dl=0"
tar -xzvf eval_checkpoints.tar.gz
rm eval_checkpoints.tar.gz

# Pseudo-GT data (385M download)
wget -O banmo_pseudo_gt.tar.gz "https://www.dropbox.com/scl/fi/08nt1z7ybu2i595uu46y4/banmo_pseudo_gt.tar.gz?rlkey=55z68f0klj32ax8br6v58mz96&dl=0"
tar -xzvf banmo_pseudo_gt.tar.gz
rm banmo_pseudo_gt.tar.gz

# Evaluation data (114M download)
wget -O ama_eval.tar.gz "https://www.dropbox.com/scl/fi/q6g78r94omaqoo8cact8f/ama_eval.tar.gz?rlkey=nxbgjsvrmpwldmi1wot9o0h80&dl=0"
tar -xzvf ama_eval.tar.gz
rm ama_eval.tar.gz
