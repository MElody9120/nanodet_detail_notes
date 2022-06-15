from abc import ABCMeta, abstractmethod

'''
    标签分配器基类
    包含回传前用于匹配正负样本的assign函数
'''
class BaseAssigner(metaclass=ABCMeta):
    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass
