import torch
import torch.nn as nn

import QLayer


class QNetwork(nn.Module):
    def __init__(self, input_vocabulary_size: int, output_vocabulary_size: int, sentence_length: int, device, q_layer_num=1):
        super().__init__()
        if q_layer_num == 1:
            self.QLayer = QLayer.QLayer(input_vocabulary_size * sentence_length, output_vocabulary_size).to(device)
        else:
            self.QLayer = QLayer.QLayer2(input_vocabulary_size * sentence_length, output_vocabulary_size).to(device)

        self.action_space_size = output_vocabulary_size
        self.device = device

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch * input_vocabulary_size * sentence_length tensor
        :param a: batch * output_vocabulary_size tensor
        :return: batch tensor
        """
        x = x.transpose(1, 2).reshape(x.size(0), -1)

        y = self.QLayer(x, a)
        return y

    def save_model(self, path: str):
        """
        保存模型参数到指定路径
        :param path: 保存文件的路径 (通常以 .pth 或 .pt 结尾)
        """
        # 推荐只保存 state_dict (参数字典)
        torch.save(self.state_dict(), path)
        print(f"Model saved successfully to {path}")

    def load_model(self, path: str):
        """
        从指定路径加载模型参数
        :param path: 参数文件的路径
        """
        try:
            # 修改处：添加 weights_only=True
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.load_state_dict(state_dict)
            print(f"Model loaded successfully from {path}")
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    @torch.inference_mode()
    def get_action(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """
        :param x:输入状态state， 二维的矩阵
        :param beta: 温度的倒数
        :return: 行动action， 一维的向量
        """
        action_q_list = torch.zeros(self.action_space_size, device=self.device)
        for i in range(self.action_space_size):
            action = torch.zeros((1, self.action_space_size), device=self.device)
            action[0, i] = 1
            action_q_list[i] = self(x.unsqueeze(0), action)[0]
        action_q_list = torch.softmax(beta * action_q_list, dim=0)
        action_index = torch.multinomial(action_q_list, num_samples=1)
        action = torch.zeros(self.action_space_size, device=self.device)
        action[action_index] = 1
        return action

    @torch.inference_mode()
    def get_value(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """
        :param x: batch * input_vocabulary_size * sentence_length tensor
        :param beta: 温度的倒数
        :return:
        """
        batch_size = x.size(0)
        q_tensor = torch.zeros((batch_size, self.action_space_size), device=self.device)
        for i in range(self.action_space_size):
            action = torch.zeros((batch_size, self.action_space_size), device=self.device)
            action[:, i] = 1
            q_tensor[:, i] = self(x, action)
        max_q, _ = torch.max(q_tensor, dim=1)
        max_q_ = max_q.unsqueeze(1).expand(batch_size, self.action_space_size)
        return max_q + torch.log(torch.sum(torch.exp(beta * (q_tensor - max_q_)), dim=1)) / beta


# if __name__ == '__main__':
#     device = torch.device("cuda")
#     test = QNetwork(sentence_length=5, input_vocabulary_size=3, output_vocabulary_size=2, device=device)
#     x = torch.randn(2, 3, 5, device=device)
#     a = torch.randn(2, 2, device=device)
#
#     print(test(x, a))
#     print(test.get_action(torch.randn(3, 5, device=device), 0.1))
