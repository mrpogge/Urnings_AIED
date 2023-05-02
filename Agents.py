import numpy as np

class Player:
    """
    class Player
        Player object constructor. It is used to create players and items which can play in the adaptive learning system

        attributes:
            user_id: str
                a string indicating the user id
            score: int
                the current number of green balls in the user's urn 
            urn_size: int
                the (current) size of the player's urn
            est: float
                the current estimate of the player's urn
            true_value: float
                the true ability of the student (used in simulation)
            sim_y: int
                helper attribute for draws from player urns
            sim_true_y: int
                helper attribute for draws from player urns (used in simulation)
            so_urn_size: int
                size of the second order urnings (used if Game_Type.adaptive_urn_type = "second order urnings")
            so_score: int
                current number of green balls in the player's second order urn
            so_est: float
                current second order estimate of the player
            container: np.array
                a numpy array containing the scores of the player over multiple games in the system
            estimate_container: np.array
                a numpy array containing the estimates of the player over multiple games in the system
            differential_container: np.array 
                    TODO: check whether it works if the adaptive urnsize algos are online
                a numpy array containing the direction of change of the player's score
            urn_container: np.array
                a numpy array containing the urn size of the player over multiple games in the system (relevant if the adaptive urn algos are online)
            so_container: np.array
                a numpy array containing the second order estimates of the player over multiple games in the system
            idx: int
                an id created when the user enters a game

        methods:
            __eq__(other: Type[Player]):
                an equivalence function which checks whether two player objects are the same
            find(id: int):
                a function with boolean output indicating whether the given player has the inputed id or not
            draw(true_score_logic:bool = False)
                a function governing the draws from urns if the true score logic is true we are drawing with the true probability (used in simulation)
            so_draw():
                a function governing the draws from the second order urns
            autocorrelation(lag: int, plots: bool = False):
                a function which calculates the autocorrelation for the chain of estimates
            so_autocorrelation(lag: int, plots: bool = False):
                a function calculating the autocorrelation for the differential container
    """
    def __init__(self, user_id: str, score: int, urn_size: int, true_value: float, so_urn_size: int = 10, stake = 16): 

        #TODO: implementing player declaration errors
        if score > urn_size:
            raise ValueError("The score can't be higher then the urn size.")

        #basic attributes
        self.user_id = user_id
        self.score = score
        self.urn_size = urn_size
        self.est = self.score/self.urn_size
        self.true_value = true_value
        self.sim_y = 8
        self.sim_true_y = 8

        #second-order urnings
        self.so_urn_size = so_urn_size
        self.so_score = int(np.round(so_urn_size / 2))
        self.so_est = self.so_score / self.so_urn_size
        
        #stakes
        self.previous_stake = stake

        #creating a container
        self.container = np.array([self.score])
        self.estimate_container = np.array([self.est])
        self.differential_container = np.array([0])
        self.urn_container = np.array([self.urn_size])
        self.so_container = np.array([self.so_est])
        self.stakes_container = np.array([self.previous_stake])

        #utility attribute
        self.idx = None
        self.scaled_score = self.score
    
    def __eq__(self, other):
        return self.user_id == other.user_id
    
    def find(self, id:int):
        return self.user_id == id
    

    def draw(self, true_score_logic: bool = False):
       #drawing the expected result based on the most frequent estimate
        if true_score_logic == False:
            sim_y = np.random.binomial(1, self.est)
            self.sim_y = sim_y
            return  sim_y
        #drawing the simulated outcome
        else:
            sim_y = np.random.binomial(1, self.true_value)
            self.sim_true_y = sim_y
            return sim_y

    def so_draw(self):
        #drawing the expected result based on the most frequent second order estimate
        sim_y = np.random.binomial(1, self.so_est)
        self.sim_y = sim_y
        return sim_y
