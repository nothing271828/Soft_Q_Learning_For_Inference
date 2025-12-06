import torch
from sympy.codegen.ast import String
import numpy as np
from Agent import Agent
from QNetwork import *
from Environment import *
import copy


class Test(object):
    def __init__(self, net1: QNetwork, env, beta, device, net2=None):
        self.net = net1
        self.net2 = net2 if net2 is not None else copy.deepcopy(net1)
        self.env = env
        self.beta = beta
        self.device = device
        pass

    def get_answers(self, question: String):
        """
        :param question:问题字符串
        :return: 一个数组，每个元素都是(state, action, reward, next_state, done)五元组
        """
        done = False
        state = self.env.string_to_tensor(question)
        answers = []
        while not done:
            action = self.net.get_action(state, self.beta)
            # 执行动作
            next_state, reward, done = self.env.step(state, action)

            answers.append((self.env.tensor_to_string(state), self.env.tensor_to_string(action.unsqueeze(1)),
                            reward, self.env.tensor_to_string(next_state), done))
            # 更新状态
            state = next_state

        return answers

    def random_test(self, number, question_length, sentence_length):
        """
        :param number: 随机测试的样本数
        :param question_length: 问题长度
        :param sentence_length: 句子长度
        :return:
        """
        answers = []
        for i in range(number):
            done = False
            state = self.env.generate_initial_state(question_length, sentence_length)
            while not done:
                action = self.net.get_action(state, self.beta)
                next_state, reward, done = self.env.step(state, action)
                state = next_state
            string = self.env.tensor_to_string(state)
            answers.append((string, sum(self.env.calculate_unmatched_brackets(string))))

        return answers

    @staticmethod
    def show_answers(answers):
        for i in answers:
            print(f"问题：{i[0]}， 网络解：{i[1]}, 奖励:{i[2]}")
        print(f"最终答案:{answers[-1][3]}")
        pass

    def compare_network_with_random_test(self, number, question_length, sentence_length):
        """
        比较两个网络在随机初态后的表现
        :param number: 随机测试的样本数
        :param question_length: 问题长度
        :param sentence_length: 句子长度
        :return:
        """
        answers1, answers2 = [], []
        for i in range(number):
            done1, done2 = False, False
            state0 = self.env.generate_initial_state(question_length, sentence_length)
            state1, state2 = state0, state0
            while not (done1 and done2):
                if not done1:
                    action1 = self.net.get_action(state1, self.beta)
                    next_state1, reward1, done1 = self.env.step(state1, action1)
                    state1 = next_state1
                if not done2:
                    action2 = self.net2.get_action(state2, self.beta)
                    next_state2, reward2, done2 = self.env.step(state2, action2)
                    state2 = next_state2
            string1, string2 = self.env.tensor_to_string(state1), self.env.tensor_to_string(state2)
            answers1.append((string1, sum(self.env.calculate_unmatched_brackets(string1))))
            answers2.append((string2, sum(self.env.calculate_unmatched_brackets(string2))))

        return answers1, answers2


if __name__ == '__main__':
    device = torch.device("cpu")
    e = DyckNTaskEnvironment(device)
    net = QNetwork(9, 7, 10, device, q_layer_num=2)
    net.load_model('model/02-3.pth')
    # net2 = QNetwork(9, 7, 10, device)
    # net2.load_model('model/01-2.pth')
    # t = Test(net, e, beta=10, device=device, net2=net2)
    t = Test(net, e, beta=20, device=device)
    ans = t.random_test(1000, 4, 10)
    for i in range(len(ans)):
        print(
            f"最终回答：{ans[i][0]}, 未补全括号及违规数:{ans[i][1]}")
    print(f"平均错误：{sum(n[1] for n in ans) / len(ans)}")

    # ans1, ans2 = t.compare_network_with_random_test(1000, 4, 10)
    # for i in range(len(ans1)):
    #     if ans1[i][0] != ans2[i][0] or ans1[i][1] > 0:
    #         print(
    #             f"网络1：最终回答：{ans1[i][0]}, 未补全括号及违规数:{ans1[i][1]}\n网络2：最终回答：{ans2[i][0]}, 未补全括号及违规数:{ans2[i][1]}")
    #         print("-" * 30)
    #
    # print(f"网络1平均错误：{sum(n[1] for n in ans1) / len(ans1)}")
    # print(f"网络2平均错误：{sum(n[1] for n in ans2) / len(ans2)}")
    # ans = t.get_answers('S([]NNNNNN')
    # t.show_answers(ans)
