# trainer.py
import torch
import os
from tqdm import tqdm

class CCoTTrainer:
    def __init__(self, config, encoder):
        self.config = config
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=config.lr)
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0)

    def train(self, multi_path_data):
        self.encoder.train()
        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for item in tqdm(multi_path_data, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                pos_paths = [p["reasoning"] for p in item["paths"] if p["is_positive"]]
                neg_paths = [p["reasoning"] for p in item["paths"] if not p["is_positive"]]

                if not pos_paths or not neg_paths:
                    continue

                anchor = pos_paths[0]
                positive = pos_paths[0]
                negative = neg_paths[0]

                a_emb, _, _ = self.encoder([anchor])
                p_emb, _, _ = self.encoder([positive])
                n_emb, _, _ = self.encoder([negative])

                loss = self.criterion(a_emb, p_emb, n_emb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(multi_path_data)
            print(f"Epoch Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(self.config.output_dir, "best_c_cot_encoder.pt")
                torch.save(self.encoder.state_dict(), save_path)
                print(f"✅ 最优模型已保存: {save_path}")