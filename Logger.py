import matplotlib as plt

class Logger(object):
    '''
        Light-weight logger for training machine learning models
    '''
    def __init__(self):
        pass

    def epoch(self):
        return Epoch()

    def append_epoch(self,epoch):
        for key, val in epoch.vars.items():
            self.append(key,val / epoch.Nbatch)

    def append(self,var,val):
        try:
            self.__getattribute__(var).append(val)
        except AttributeError:
            self.__setattr__(var,[])
            self.__getattribute__(var).append(val)

    def set(self,var,val):
        self.__setattr__(var,val)

    def add(self,var,val=1):
        try:
            variable = self.__getattribute__(var)
            variable = variable + val
        except AttributeError:
            self.__setattr__(var,val)

    def plot_list(self,var_list,ax=None):
        for var in var_list:
            self.plot(var,ax=ax,label=var)

    def plot(self,var,ax=None,label=''):
        try:
            variable = self.__getattribute__(var)
        except AttributeError:
            raise AttributeError('No log for variable "' + var + '"')

        if(ax is None):
            plt.plot(variable,label=label)
        else:
            ax.plot(variable,label=label)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __repr__(self):
        return str(list(self.__dict__))
