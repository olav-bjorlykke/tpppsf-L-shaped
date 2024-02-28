
class LShapedAlgoritm:
    def __init__(self) -> None:
        pass
'''
    : Init and solve master problem w/o optimality cut
    : Init subproblems based on master problem solution
        : Solve SPs
        : Add optimality cut to MP
        : Solve master problem
        : Pass MP values to SPs
        : Return to step 3 if MP solution != previous MP solution
        : Terminate
'''