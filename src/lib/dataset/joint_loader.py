import random

class JointIterator:
    def __init__(self, iter1, iter2, dataset1, dataset2):
        self.iter1 = iter1
        self.iter2 = iter2
        self.num_steps = [5, 5]
        self.loader_ind = 0
        self.counter = self.num_steps[self.loader_ind]
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __next__(self):
        if self.counter == 0:
            ind = random.randint(0, 1)
            self.loader_ind = ind
            self.counter = self.num_steps[ind]

        if self.loader_ind == 0:
            if type(self.iter1) == list:
                dataset_ind = random.randint(0, len(self.iter1) - 1)
                iter1 = self.iter1[dataset_ind]
                result = next(iter1, None)
                if result is None or len(result[0]['image']) != self.dataset1[dataset_ind].batch_size:
                    iter1 = iter(self.dataset1[dataset_ind])
                    result = next(iter1, None)
                    self.iter1[dataset_ind] = iter1
            else:
                result = next(self.iter1, None)
                if result is None or len(result[0]['image']) != self.dataset1.batch_size:
                    self.iter1 = iter(self.dataset1)
                    result = next(self.iter1, None)
        else:
            result = next(self.iter2, None)
            if result is None:
                self.iter2 = iter(self.dataset2)
                result = next(self.iter2, None)

        self.counter -= 1

        return result

    # def __next__(self):
    #     if random.randint(0, 1) == 0:
    #         result = next(self.iter1, None)
    #         if result is None:
    #             self.iter1 = iter(self.dataset1)
    #             result = next(self.iter1, None)
    #             # result = next(self.iter2, None)
    #             if result is None:
    #                 raise StopIteration
    #     else:
    #         result = next(self.iter2, None)
    #         if result is None:
    #             # result = next(self.iter1, None)
    #             self.iter2 = iter(self.dataset2)
    #             result = next(self.iter2, None)
    #             if result is None:
    #                 raise StopIteration

    #     return result

class JointLoader:

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        if type(dataset1) == list:
            self.dataset = dataset1[0].dataset
        else:
            self.dataset = dataset1.dataset

    def __iter__(self):
        if type(self.dataset1) == list:
            iter1 = []
            for dts in self.dataset1:
                iter1.append(iter(dts))
        else:
            iter1 = iter(self.dataset1)
        return JointIterator(iter1, iter(self.dataset2), self.dataset1, self.dataset2)

    def __len__(self):
        len1 = 0
        if type(self.dataset1) == list:
            for dts in self.dataset1:
                len1 += len(dts)
            else:
                len1 = len(self.dataset1)
        return len1 + len(self.dataset2)
