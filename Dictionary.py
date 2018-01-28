import jieba


class Dictionary:
    def __init__(self):
        self.word_count = {}
        self.word_idex = {}
        self.idex_word = {}
        with open("data/humanwrite.txt") as fin:
            for line in fin.readlines():
                ans = jieba.cut(line.split()[1])
                for each in ans:
                    if each in self.word_count:
                        self.word_count[each] = self.word_count[each] + 1
                    else:
                        self.word_count[each] = 1

        with open("data/humantranslate.txt") as fin:
            for line in fin.readlines():
                ans = jieba.cut(line.split()[1])
                for each in ans:
                    if each in self.word_count:
                        self.word_count[each] = self.word_count[each] + 1
                    else:
                        self.word_count[each] = 1

        with open("data/google.txt") as fin:
            for line in fin.readlines():
                ans = jieba.cut(line.split()[1])
                for each in ans:
                    if each in self.word_count:
                        self.word_count[each] = self.word_count[each] + 1
                    else:
                        self.word_count[each] = 1

        with open("data/baidu.txt") as fin:
            for line in fin.readlines():
                ans = jieba.cut(line.split()[1])
                for each in ans:
                    if each in self.word_count:
                        self.word_count[each] = self.word_count[each] + 1
                    else:
                        self.word_count[each] = 1

        word_count = [(k, self.word_count[k]) for k in sorted(self.word_count, key=self.word_count.get, reverse=True)]

        print("the text has {} tokens".format(word_count.__len__()))
        count = 0
        for each in word_count:
            if (each[1] >= 5):
                self.idex_word[count] = each[0]
                self.word_idex[each[0]] = count
                count = count + 1

        self.idex_word[count] = "UNKNOWN"
        self.word_idex["UNKNOWN"] = count
        count = count + 1

        self.idex_word[count] = "EOF"
        self.word_idex["EOF"] = count

        print("Dictionary has {} tokens".format(count + 1))


Dictionary()
