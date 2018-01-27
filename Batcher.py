class Batcher(object):
    '''
        Wrapper to simplify tracking scalars during batch training
    '''
    def __init__(self):
        self.Nbatch = 0
        self.vars = {}

    def batch(self,batch_size):
        '''
            Call at the beginning of each batch
        '''
        self.Nbatch = self.Nbatch + 1

    def add(self,var,val):
        '''
            Add value 'val' to scalar 'var'. If 'var' is not yet tracked
            set val = var

            Params:
            val (String): scalar variable to add var to
            var (scalar): value to add to var
        '''
        try:
            self.vars[var] = self.vars[var] + val

        except KeyError:
            self.vars[var] = val

    def report(self):
        for key, val in self.vars.items():
            print(key + ": %.4e, " % (val / self.Nbatch,),end='')
        print()

    def write(self,summary_writer,epoch):
        for key, val in self.vars.items():
            summary_writer.add_scalar(key,val / self.Nbatch,epoch)
        self.reset()

    def reset(self):
        self.vars = {}
        self.Nbatch = 0
