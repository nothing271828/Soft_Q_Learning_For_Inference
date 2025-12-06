import torch
import random

from sympy.physics.units import length


class DyckNTaskEnvironment:
    def __init__(self, device):
        self.device = device
        self.vocabulary = {}
        self.vocabulary_list = ['E', '(', ')', '[', ']', '{', '}', 'S', 'N']
        n = len(self.vocabulary_list)
        for i, word in enumerate(self.vocabulary_list):
            one_hot = [0 for _ in range(n)]
            one_hot[i] = 1
            self.vocabulary[word] = torch.tensor(one_hot, device=self.device).unsqueeze(1)

        self.inverse_vocabulary = {i: word for i, word in enumerate(self.vocabulary_list)}

        self.left_brackets = ['(', '{', '[']
        self.right_brackets = [')', '}', ']']
        self.match = {')': '(', '}': '{', ']': '['}

    def string_to_tensor(self, string):
        vectors = [self.vocabulary[char] for char in string]
        result_tensor = torch.cat(vectors, dim=1)
        return result_tensor

    def tensor_to_string(self, tensor):
        chars = []
        for i in range(tensor.shape[1]):
            one_hot = tensor[:, i]
            index = torch.argmax(one_hot).item()
            char = self.inverse_vocabulary[index]
            chars.append(char)
        return ''.join(chars)

    def generate_initial_state(self, question_length, sentence_length):
        """
        :param question_length: 初始给定的字符串数量
        :param sentence_length: 句子总长度
        :return:
        """
        assert question_length <= sentence_length
        state = ['N'] * sentence_length

        # S as start
        state[0] = 'S'
        stack = []

        # Fill index 1 ~ question_length-1
        for i in range(1, question_length):
            # 随机选择填入左或右括号
            if random.random() < 0.7 or not stack:
                # 只能写左括号或栈空时强制左括号
                bracket = random.choice(self.left_brackets)
                state[i] = bracket
                stack.append(bracket)
            else:
                # 尝试写一个合法右括号
                # 找到与当前栈顶匹配的右括号
                left_top = stack[-1]
                right = None
                for r, l in self.match.items():
                    if l == left_top:
                        right = r
                        break
                # 写入右括号并出栈
                state[i] = right
                stack.pop()

        state = ''.join(state)
        return self.string_to_tensor(state)

    def step(self, state, action):
        """
        :param state: 输入状态
        :param action: 输入此时的行动
        :return: next_state, reward, done 下一个状态，奖励和是否达到末态
        """
        string = self.tensor_to_string(state)
        char = self.tensor_to_string(action.unsqueeze(1))
        next_string = self.add_char(string, char)
        if 'N' not in next_string or 'E' in next_string:
            done = True
            reward = -sum(self.calculate_unmatched_brackets(next_string))
        else:
            done = False
            before, latter = self.calculate_unmatched_brackets(string), self.calculate_unmatched_brackets(next_string)
            change = (-before[0] + latter[0], -before[1] + latter[1])
            reward = -change[1] + (-change[0]*0.2 if change[0] >= 0 else -change[0])

        next_state = self.string_to_tensor(next_string)
        return next_state, reward, done

    def calculate_unmatched_brackets(self, string):
        stack = []
        rule_count = 0
        counting = False
        for char in string:
            if not counting:
                if char == 'S':
                    counting = True
            else:
                if char in self.left_brackets:
                    stack.append(char)
                elif char in self.right_brackets:
                    if stack:
                        if self.match[char] == stack[-1]:
                            stack.pop()
                        else:
                            rule_count += 1
                    else:
                        rule_count += 1

        return len(stack), rule_count

    @staticmethod
    def add_char(string, char):
        for i, s in enumerate(string):
            if s == 'N':
                return string[:i] + char + string[i + 1:]
        return string


if __name__ == '__main__':
    pass
