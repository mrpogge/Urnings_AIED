import numpy as np
from typing import Optional, Type
from Game_Type import Game_Type
from Agents import Player

class Urnings:
    """
    class Urnings: 
        A framework which saves all the chosen game options and let the Players and Items play the Urnings Game.
        Both simulation and Data analysis moduls are available.

        attributes: 
            players: list[Player]
                list of Player objects we would like to analyse, representing the students/players in the adaptive learning system
                For details see Player.__doc__().
            items: list[Player]
                list of Player objects we would like to analyse, representing the items in the adaptive learnings system.
                For details see Player.__doc__().
            game_type: Game_Type
                A Game_Type object which save all the options we want to declare before we start the learning system. 
                For details see Game_Type.__doc__()
            data: AlsData
                An AlsData object which contains the serialized, and restructured data, it enables us to create the Player objects and a punchcard which 
                governs the data's item selection procedure. 
                For details see Als Data.__doc__()
            queue_pos: dict
                Dictionary with all the defined item's user id as key. Used in the paired update system. This dictionary contains the items waitning for a positive update.
            queue_neg_ dict
                Dictionary with all the defined item's user id as key. Used in the paired update system. This dictionary contains the items waitning for a negative update.
            adaptive_matrix: np.ndarray
                A numpy array saving the probability of selection of each player item paires (TODO: Refactor it into an Urn based matrix for better computation performance)
            game_count: int
                The number of games we played in the system.
            item_green_balls: list[int]
                A list of integers saving the number of green balls in all item urns. It can be used to check whether there is green ball inflation which is against the 
                assumption of Urnings model. Can be corrected by setting Game.Type.paired_update = True.
            prop_correct: np.ndarray
                A numpy array saving the proportion of correct responses in each player and item urnings. It can be used to investigate model fit by plotting it against 
                the true model predictions using contour plots.
            number_bin_correct: np.ndarray
                A numpy array saving the number of responese in each play and item urnings.
            fit_correct:  np.ndarray
                A numpy array saving the model implied proportions. It can be used to investigate model fit.
        
        methods:
            adaptive_rule_normal(self)
                calculates the probability matrix for adaptive item selection for all player item pairs using the normal method, for details see. Bolsinova et al (2022).
            adaptive_rule_partial(self, player: Type[Player], item: Type[Player])
                helper function for more optimal calculation of the probability matrix for adaptive item selection, by only updating the given column and row based on the 
                player and the item id
            matchmaking(self, ret_adaptive_matrix: bool = False)
                function governing the matchmaking by using either the adaptive or the nonadaptive alternatives, it can retrun the updated probability matrix for adaptive
                item selection
            urnings_game(player: Type(Player), item: Type(Player))
                The summary function which set's up the game environment. It activates after item selection and updates the item and player properties
            play(n_games: int, test: bool = False)
                The function which starts the Urnings game. Test can be used to let each player play the same amount of games. This feature can be useful with simulation studies



    """
    def __init__(self, players: list[Type[Player]], items: list[Type[Player]], game_type: Type[Game_Type], control_draws = 3):
        # initial data for the Urnings frameweok
        self.players = players
        self.items = items
        self.game_type = game_type

        #initialsing idexes
        for pl in range(len(self.players)):
            self.players[pl].idx = pl
        
        for it in range(len(self.items)):
            self.items[it].idx = it

        #containers for the paired update queue
        self.queue_pos = {k.user_id : 0 for k in self.items}
        self.queue_neg = {k.user_id : 0 for k in self.items}

        #helper attribute for the paired update queue
        sum_gb_init = 0
        for it in self.items:
            sum_gb_init += it.score

        sum_gb_init_all = 0
        for pl in self.players:
            sum_gb_init_all += pl.score

        sum_gb_init_all += sum_gb_init        
        self.item_green_balls = [sum_gb_init]
        self.total_green_balls = [sum_gb_init_all]

        sum_total_init = 0
        for it in self.items:
            sum_total_init += it.urn_size
        for pl in self.players:
            sum_total_init += pl.urn_size

        self.total_num_balls = [sum_total_init]

        #arrays to calculate model fit
        if self.game_type.adaptive_urn == False:
            self.game_type.max_urn = self.players[0].urn_size

        self.prop_correct = np.zeros((self.game_type.max_urn +1, self.items[0].urn_size+1))
        self.number_per_bin = np.zeros((self.game_type.max_urn +1, self.items[0].urn_size+1))
        self.fit_correct = np.zeros((self.game_type.max_urn +1, self.items[0].urn_size+1))
        self.adaptive_correct = np.zeros((self.game_type.max_urn +1, self.items[0].urn_size + 1))
        
        #helper attributes for the adaptive item selection
        for pl in self.players:
            pl.scaled_score = int(pl.score * (self.game_type.max_urn / pl.urn_size))

        self.adaptive_matrix_binned = np.zeros((self.game_type.max_urn + 1, self.items[0].urn_size + 1))
        for p in range(self.game_type.max_urn + 1):
            for i in range(self.items[0].urn_size + 1):
                self.adaptive_matrix_binned[p,i] = self.normal_method_helper(p, i, self.game_type.max_urn, self.items[0].urn_size)
        
        self.item_bins = {str(i):[] for i in range(self.items[0].urn_size + 1)}
        for it in self.items:
            bin_idx = str(int(it.score))
            self.item_bins[bin_idx].append(it)
        
        #helper attribute for data analysis
        self.game_count = 0

        #helper for dev
        self.bugfix = 0

        self.control_draws = control_draws

    
    def normal_method_helper(self, R_i, R_j, n_i, n_j):
        return np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)        
    
    def matchmaking(self, player_id: Optional[int] = None):
        if self.game_type.adaptivity == "n_adaptive":
            if player_id is None:
                player_id = np.random.randint(0,len(self.players))
            item_id = np.random.randint(0, len(self.items))

            return self.players[player_id], self.items[item_id]
        
        elif self.game_type.adaptivity == "adaptive":
            if player_id is None:
                player_id = np.random.randint(0,len(self.players))
            #calculating normalising constant
            player_scaled_score = self.players[player_id].scaled_score
            selected_item_bins = self.adaptive_matrix_binned[player_scaled_score, :]
            num_per_bin = [len(self.item_bins[str(i)]) for i in range(self.items[0].urn_size + 1)]

            item_probs_unnormalised = []
            for sib, npb in zip(selected_item_bins, num_per_bin):
                for b in range(npb):
                    item_probs_unnormalised.append(sib)
            
            item_id_list = []
            for ib in range(self.items[0].urn_size + 1):
                binned_item_list = self.item_bins[str(ib)]
                for bil in binned_item_list:
                    item_id_list.append(bil.idx)
                    
            item_probs_normalised = np.array(item_probs_unnormalised) / np.sum(np.array(item_probs_unnormalised))
            item_id = np.random.choice(item_id_list, p=item_probs_normalised)
            item = self.items[item_id]
            
            return self.players[player_id], item
        
    def urnings_game(self, player: Type[Player], item: Type[Player]):
        self.adaptive_correct[player.scaled_score, item.score] += 1
        #--------------------------------------calculate the estimated response-----------------------------------------#

        result, expected_results = self.game_type.draw_rule(player, item) 
        
        #--------------------------------------update the urnings -----------------------------------------------------#
        if self.game_type.adaptive_urn_type == "stakes_second_order_urnings" or self.game_type.adaptive_urn_type == "stakes_permutation":
            player_stake = self.game_type.calculate_stakes(player, item, self.control_draws)
            player_proposal, item_proposal = self.game_type.updating_with_stakes(player, item, result, expected_results, player_stake)
        elif self.game_type.adaptive_urn_type == "fixed_stakes":
            player_proposal, item_proposal = self.game_type.updating_with_stakes(player, item, result, expected_results, player.previous_stake)
        else:
            player_proposal, item_proposal = self.game_type.updating_rule(player, item, result, expected_results)


        #--------------------------------------calculate metropolis correction————————————————————————————————————————–#
        
        #correction for adaptivity
        adaptivity_corrector = self.game_type.adaptivity_correction(player, item, player_proposal, item_proposal, self.adaptive_matrix_binned, self.item_bins)

        #correction for algorithm type
        try:
            metropolis_corrector = self.game_type.metropolis_correction(player, item, player_proposal, item_proposal)
        except:
            metropolis_corrector = 1
        
        acceptance = min(1, metropolis_corrector * adaptivity_corrector)
        u = np.random.uniform()

        #save the   values for later methods
        player_prev = player.score
        item_prev = item.score

        #metropolis step
        if u < acceptance:
            self.bugfix += 1
            player.score = player_proposal
            player.scaled_score = int(player.score * (self.game_type.max_urn / player.urn_size))
            item.score = item_proposal
            player.est = player.score / player.urn_size
            item.est = item.score / item.urn_size

        #--------------------------------------Paired Item Update-----------------------------------------------------#
        #calculating the difference
        player_diff = player.score - player_prev
        item_diff = item.score - item_prev

        if player_diff > 1:
            player_diff = 1
        elif player_diff < -1:
            player_diff = -1
        
        if item_diff > 1:
            item_diff = 1
        elif item_diff < -1:
            item_diff = -1

              
        self.game_type.paired_update(item, self.items, item_diff, self.queue_neg, self.queue_pos)

        #track the changes in the proportion of green balls in the whole system 

        #-------------------------------------Place items in a new bin after the updating is done---------------------#
        #TODO: Optimise this
        #adaptive item place recalculation
        self.item_bins = {str(i):[] for i in range(self.items[0].urn_size + 1)}
        for it in self.items:
            bin_idx = str(it.score)
            self.item_bins[bin_idx].append(it)
        

        #------------------------------------Save data before the adaptive urn change algos---------------------------#
        #appending new update to the container
        player.container = np.append(player.container, player.score)
        item.container = np.append(item.container, item.score)

        player.estimate_container = np.append(player.estimate_container, player.est)
        item.estimate_container = np.append(item.estimate_container, item.est)

        #appending second order results
        player.differential_container = np.append(player.differential_container, player_diff)
        item.differential_container = np.append(item.differential_container, item_diff)

        #--------------------------------------Adaptive urn change algos----------------------------------------------#
        #Second Order Urnings
        self.game_type.second_order_urnings(player, player_diff)
 
        #saving the second order urnings
        player.so_container = np.append(player.so_container, player.so_est)
        

        #-------------------------------------Adaptive urn_size------------------------------------------------------#
        self.game_type.adaptive_urn_change(player, control_draws=self.control_draws)
        player.scaled_score = int(player.score * (self.game_type.max_urn / player.urn_size))

        #saving urnings values
        player.urn_container = np.append(player.urn_container, player.urn_size)
        item.urn_container = np.append(item.urn_container, item.urn_size)
        player.stakes_container = np.append(player.stakes_container, player.previous_stake)


        #------------------------------------evaluating fit---------------------------------------------------------#
        #self.prop_correct[player.scaled_score, item.score] += result
        #self.number_per_bin[player.scaled_score, item.score] += 1
        #self.fit_correct[player.scaled_score, item.score] += expected_results
         
            

    def play(self, n_games: int, test: bool = False):
        for ng in range(n_games):
            if test == True:
                for pl in range(len(self.players)):
                    current_player, current_item = self.matchmaking(pl)
                    self.urnings_game(current_player, current_item)

                    #calculating the number of green balls in the item urns
                    # sum_total = 0
                    # sum_gb = 0
                    # for it in self.items:
                    #     sum_gb += it.score
                    #     sum_total += it.urn_size
                    
                    # sum_gb_all = 0
                    # for pl in self.players:
                    #     sum_gb_all += pl.score
                    #     sum_total += pl.urn_size
                    
                    # sum_gb_all += sum_gb
                    
                    # self.item_green_balls.append(sum_gb)
                    # self.total_green_balls.append(sum_gb_all)
                    # self.total_num_balls.append(sum_total)
                if ng % 10 == 0:
                    print(ng)
            else:
                current_player, current_item = self.matchmaking()
                self.urnings_game(current_player, current_item)
            self.game_count += 1
                



    