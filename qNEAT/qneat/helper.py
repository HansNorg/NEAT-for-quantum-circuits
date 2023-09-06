class InnovationNumber(object):
    '''
    Class for keeping a global innovation number.
    Inspired by QMS-Xenakis
    '''
    def __init__(self):
        '''
        Initialise the innovation generator, start at 0.
        '''
        self._innovation_number:int = 0

    def next(self):
        '''
        Get the next innovation number.
        Increments the innovation number.
        '''
        self._innovation_number += 1
        return self._innovation_number
    
    # def __new__(cls):
    #     '''
    #     Make sure there cannot be multiple InnovationNumber instances (Singleton).
    #     '''
    #     if not hasattr(cls, 'instance'):
    #         cls.instance = super(InnovationNumber, cls).__new__(cls)
    #     return cls.instance