import numpy as np
from typing import Optional, Type
from Agents import Player
import utilities as util

class Game_Type:
    """
    class Game_Type:
        a class to setup the different analysis and simulation options for the game environment, ensures the modular architecture

        attributes:
            adaptivity: str
                takes values from ["n_adaptive", "adaptive"], governs the item selection
            alg_type: str
                takes values from ["Urnings1", "Urnings2"], choses the type of algorithm to use
            updating_type: str
                currently only one_dimensional TODO: implement multidimensional, response time and polytomous urning and ELO rating system
            paired_update: bool
                indicates the use of paired update
            adaptive_urn: bool
                indicates the use of adaptive urn algorithms
            adaptive_urn_type: str
                takes values from ["permutation", "second_order_urnings"], choses the type of adaptive urn size algorithm
            min_urn: int
                the minimum urn value (used in adaptive urn size algorithms)
            max_urn: int
                the maximim urn value (used in adaptive urn size algorithms)
            freq_change: int
                the frequency (iteration) to change the urn size (used in adaptive_urn_type = "permutation" witn permutation_test = False)
            window: int
                the size of the moving window (used in adaptive_urn_type = "permutation")
            bound: int
                the number of wins which is considered a winstreak (used in adaptive_urn_type = "permutation" witn permutation_test = False)
            permutation_test: bool
                indicates whether we use permutation test or not
            n_permutations: int
                indicates the number of permutations generated if window < 15 the algorithm uses exact test
            perm_p_val: float
                p value for the permutation test

    """
    def __init__(self, 
                adaptivity: str,
                alg_type: str, 
                updating_type: str = "one_dim", 
                paired_update: bool = False, 
                adaptive_urn: bool = False,
                adaptive_urn_type: Optional[str] = None, 
                min_urn: Optional[int] = None,
                max_urn: Optional[int] = None,
                min_stakes: int = 1,
                max_stakes: Optional[int] = None,
                multiplication_factor: Optional[int] = 1,
                freq_change: Optional[int] = None,
                window: int = 2,
                bound: Optional[int] = None,
                permutation_test: bool = False,
                n_permutations: int = 1000,
                perm_p_val: float = 0.05):

        self.adaptivity = adaptivity
        self.alg_type = alg_type
        self.updating_type = updating_type
        self.item_pair_update = paired_update
        self.adaptive_urn = adaptive_urn
        self.adaptive_urn_type = adaptive_urn_type
        self.min_urn = min_urn
        self.max_urn = max_urn
        self.min_stakes = min_stakes
        self.max_stakes = max_stakes
        self.multiplication_factor = multiplication_factor
        self.freq_change = freq_change
        self.window = window
        self.bound = bound
        self.permutation_test = permutation_test
        self.n_permutations = n_permutations
        self.perm_p_val = perm_p_val

        #container for updates
        self.queue_pos = []
        self.queue_neg = []

        #all combinations for exact permutation test
        self.all_comb = util.all_binary_combination(self.window)
    
    def draw_rule(self, player: Type[Player], item: Type[Player]):
        
        #urnings 1 algorithm 
        if self.alg_type == "Urnings1":
            #simulating the observed value
            while player.sim_true_y == item.sim_true_y:
                player.draw(true_score_logic = True)
                item.draw(true_score_logic = True)

            result = player.sim_true_y
            player.sim_true_y = item.sim_true_y = 8


            #calculating expected score
            counter = 0
            while player.sim_y == item.sim_y and counter < 1000:
                player.draw()
                item.draw()
                counter += 1

            expected_results = player.sim_y
            player.sim_y = item.sim_y = 8
            counter = 0

        #urnings 2 algorithm
        elif self.alg_type == "Urnings2":
            #simulating the observed value
            while player.sim_true_y == item.sim_true_y:
                player.draw(true_score_logic = True)
                item.draw(true_score_logic = True)
            
            result = player.sim_true_y
            player.sim_true_y = item.sim_true_y = 8

            #calculating expected value
            player.est = (player.score + result) / (player.urn_size + 1)
            item.est = (item.score + 1 - result) / (item.urn_size + 1)

            while player.sim_y == item.sim_y:
                    player.draw()
                    item.draw()
                
            expected_results = player.sim_y
            player.sim_y = item.sim_y = 8
            
            #returning to the original urn conig
            player.est = player.score / player.urn_size
            item.est = item.score / item.urn_size

        return result, expected_results
  
    def updating_rule(self, 
                      player: Type[Player], 
                      item: Type[Player], 
                      result: int, expected_results: int):
        
        if self.updating_type == "one_dim":
            #updating scores
            player_prop = player.score  + result - expected_results
            item_prop = item.score  + (1 - result) - (1 - expected_results)

            #Making sure that the urnsize is bigger than the total number of balls obviously
            if player_prop > player.urn_size:
                player_prop = player.urn_size
                
            if player_prop < 0:
                player_prop = 0
                
            if item_prop > item.urn_size:
                item_prop = item.urn_size
                
            if item_prop < 0:
                item_prop = 0
            
        return player_prop, item_prop
    
    def calculate_stakes(self, player:Type[Player], item:Type[Player], control_draws):
        if self.adaptive_urn_type == "stakes_permutation":
            if len(player.differential_container) >= self.window:
                conv_stat = player.differential_container[-self.window:]
                permute_means = np.mean(self.all_comb * conv_stat, axis=1)
                p_value = 1 - np.sum(permute_means < np.abs(np.mean(conv_stat)))/len(permute_means)
                
                if p_value < self.perm_p_val:
                    player.previous_stake = self.max_stakes
                elif player.previous_stake != 1:
                    player.previous_stake = int(player.previous_stake / 2 ) 

            return player.previous_stake 
                
        elif self.adaptive_urn_type == "stakes_second_order_urnings":
            if len(player.so_container) >= self.window and len(player.so_container) % self.window == 0:
                    draw_urn_control = np.sum(np.random.binomial(1, np.mean(player.so_container[-self.window:]), control_draws))
                    if (draw_urn_control == control_draws or draw_urn_control == 0) and player.previous_stake > self.min_stakes :
                        player.previous_stake = self.max_stakes

                    elif player.previous_stake != 1:
                        player.previous_stake = player.previous_stake - 1
            return player.previous_stake
                    

    def updating_with_stakes(self, player, item, result, expected_results, stakes_player):
        stake = stakes_player

        is_out_bounds_player = ((player.score + stake) > player.urn_size) or ((player.score - stake) < 0)
        is_out_bounds_item = ((item.score + stake) > item.urn_size) or ((item.score - stake) < 0)

        if is_out_bounds_player == True or is_out_bounds_item == True:
            player_prop, item_prop = self.updating_rule(player,item,result,expected_results)
        else:
            #updating scores
            player_prop = player.score  + stake * (result - expected_results)
            item_prop = item.score  + stake * ((1 - result) - (1 - expected_results))

            #Making sure that the urnsize is bigger than the total number of balls obviously
            if player_prop > player.urn_size:
                player_prop = player.urn_size
                
            if player_prop < 0:
                player_prop = 0
                
            if item_prop > item.urn_size:
                item_prop = item.urn_size
                
            if item_prop < 0:
                item_prop = 0
            
        return player_prop, item_prop
            

    def metropolis_correction(self, player: Type[Player], item: Type[Player], player_proposal: int, item_proposal: int):
        
        #algorithm type to provide the first part of the metropolis correction 
        if self.alg_type == "Urnings1":
            old_score = player.score * (player.urn_size - item.score) + (item.urn_size - player.score) * item.score
            new_score = player_proposal * (player.urn_size - item_proposal) + (item.urn_size - player_proposal) * item_proposal

            metropolis_corrector = old_score/new_score
        
        elif self.alg_type == "Urnings2":
            
            metropolis_corrector = 1
        
        return metropolis_corrector

    def adaptivity_correction(self, player: Type[Player], item: Type[Player], player_proposal: int, item_proposal: int, adaptive_matrix_binned: np.ndarray, item_bins: dict):
        if self.adaptivity == "adaptive":
            num_per_bin = [len(item_bins[str(i)]) for i in range(item.urn_size + 1)]
            current_selection_prob = adaptive_matrix_binned[player.scaled_score, item.score] / np.sum(adaptive_matrix_binned[player.scaled_score, :] * num_per_bin)

            new_num_per_bin = num_per_bin
            new_num_per_bin[item.score] -= 1
            new_num_per_bin[item_proposal] += 1
            player_proposal_scaled = int(player_proposal * (self.max_urn / player.urn_size))
            proposed_selection_prob = adaptive_matrix_binned[player_proposal_scaled, item_proposal] / np.sum(adaptive_matrix_binned[player_proposal_scaled, :] * new_num_per_bin)
            adaptivity_corrector = proposed_selection_prob/current_selection_prob
            
        else:
            adaptivity_corrector = 1
        
        return adaptivity_corrector

    def paired_update(self, item: Type[Player], items: list[Type[Player]], item_diff: int, queue_neg: dict, queue_pos: dict):
        if self.item_pair_update == True:
            if item_diff == 1:
                if all(i == 0 for i in list(queue_neg.values())):
                    queue_pos[item.user_id] += 1
                    if item.score > 0:
                        item.score -= 1
                        item.est = item.score / item.urn_size 
                else:
                    candidates = {k:v for k,v in queue_neg.items() if v >= 1}
                    idx = np.random.randint(0, len(candidates.keys()))
                    candidate_user_id = list(candidates)[idx]
                    
                    for it in items:
                        if it.user_id == candidate_user_id:
                            candidate_item = it 
                    
                    counter = 0
                    while candidate_item.score <= 0:
                        candidates = {k:v for k,v in queue_neg.items() if v >= 1}
                        idx = np.random.randint(0, len(candidates.keys()))
                        candidate_user_id = list(candidates)[idx]
                    
                        for it in items:
                            if it.user_id == candidate_user_id:
                                candidate_item = it
                        counter +=1 
                        if counter > 100:
                            break
                        
                    queue_neg[candidate_user_id] = 0
                    candidate_item.score -= 1
                    candidate_item.est = candidate_item.score / candidate_item.urn_size

            elif item_diff == -1:
                if all(i == 0 for i in list(queue_pos.values())):
                    queue_neg[item.user_id] += 1
                    if item.score < item.urn_size:
                        item.score += 1
                        item.est = item.score / item.urn_size 
                else:
                    candidates = {k:v for k,v in queue_pos.items() if v >= 1}
                    idx = np.random.randint(0, len(candidates.keys()))
                    candidate_user_id = list(candidates)[idx]

                    for it in items:
                        if it.user_id == candidate_user_id:
                            candidate_item = it
                    
                    counter = 0
                    while candidate_item.score >= candidate_item.urn_size:
                        candidates = {k:v for k,v in queue_pos.items() if v >= 1}
                        idx = np.random.randint(0, len(candidates.keys()))
                        candidate_user_id = list(candidates)[idx]

                        for it in items:
                            if it.user_id == candidate_user_id:
                                candidate_item = it
                        counter += 1
                        if counter > 100:
                            break
                    
                    queue_pos[candidate_user_id] = 0
                    candidate_item.score += 1
                    candidate_item.est = candidate_item.score / candidate_item.urn_size

    def second_order_urnings(self, player: Type[Player], player_diff: int):
        so_diff = player_diff
        if so_diff == -1:
            so_diff = 0

        expected_result = np.random.binomial(1, player.so_est, 1)

        player.so_score = player.so_score + so_diff - expected_result

        player.so_est = player.so_score / player.so_urn_size
    

          
    def adaptive_urn_change(self, player: Type[Player], control_draws = 2):
        
        if self.adaptive_urn == True:
            if self.adaptive_urn_type == "permutation":
                if len(player.differential_container) >= self.window and len(player.differential_container) % self.window == 0:
                    conv_stat = player.differential_container[-self.window:]

                    if self.permutation_test == False:
                        #check the stats
                        if np.sum(conv_stat) >= self.bound and player.urn_size > self.min_urn:
                            change = player.urn_size / self.min_urn
                            player.score = int(np.round(player.score / change))
                            player.urn_size = self.min_urn
                            player.est = player.score / player.urn_size
                        elif len(player.differential_container) % self.freq_change == 0 and player.urn_size < self.max_urn:
                            player.urn_size = player.urn_size * 2
                            player.score =  player.score * 2
                            player.est = player.score / player.urn_size
                    else:
                        permute_means = np.mean(self.all_comb * conv_stat, axis=1)
                        p_value = 1 - np.sum(permute_means < np.abs(np.mean(conv_stat)))/len(permute_means)

                        if p_value < self.perm_p_val:
                            change = player.urn_size / self.min_urn
                            player.score = int(np.round(player.score / change))
                            player.urn_size = self.min_urn
                            player.est = player.score / player.urn_size
                        elif player.urn_size < self.max_urn:
                            player.urn_size = player.urn_size * 2
                            player.score =  player.score * 2
                            player.est = player.score / player.urn_size

            elif self.adaptive_urn_type == "second_order_urnings":
                if len(player.so_container) >= self.window and len(player.so_container) % self.window == 0:
                    draw_urn_control = np.sum(np.random.binomial(1, np.mean(player.so_container[-self.window:]), control_draws))
                    if (draw_urn_control == control_draws or draw_urn_control == 0) and player.urn_container[-1] > self.min_urn:
                        change = player.urn_size / self.min_urn
                        player.score = int(np.round(player.score / change))
                        player.urn_size = self.min_urn
                        player.est = player.score / player.urn_size
                    
                    elif player.urn_container[-1] < self.max_urn:
                        player.urn_size = player.urn_container[-1] * 2
                        player.score = player.score * 2
                        player.est = player.score / player.urn_size
