import abc


class BaseSet(abc.ABC):
    @abc.abstractmethod
    def sample(self, initial_position, expl_model):
        del initial_position
        raise NotImplementedError