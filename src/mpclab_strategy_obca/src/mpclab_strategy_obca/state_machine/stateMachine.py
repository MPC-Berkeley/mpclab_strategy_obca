#!/usr/bin python3

import numpy as np
import numpy.linalg as la

from mpclab_strategy_obca.utils.types import experimentParams
from mpclab_strategy_obca.utils.types import experimentStates

state_num_dict = {'Start' : -1,
    'End' : -2,
    'Free-Driving' : 0,
    'Safe-Confidence' : 1,
    'Safe-Yield' : 2,
    'Safe-Infeasible' : 3,
    'HOBCA-Unlocked' : 4,
    'HOBCA-Locked' : 5,
    'Emergency-Break' : 6}

num_state_dict = {v: k for k, v in state_num_dict.items()}

class stateMachine(object):

    def __init__(self, params=experimentParams()):
        self.state = "Start" # current state
        self.strategy = "N/A" # Current strategy

        self.N = params.N
        self.dt = params.dt

        self.collision_buffer_r = params.collision_buffer_r
        self.confidence_thresh = params.confidence_thresh
        self.x_max = params.x_max
        self.T = params.T
        self.T_tv  = params.T_tv
        self.lock_steps = params.lock_steps
        self.strategy_names = params.strategy_names

        self.a_lim = params.a_lim

    def state_transition(self, exp_states=experimentStates()):
        """
        Transition between states
        """
        state_next = "N/A"
        strategy_next = "N/A"

        t = exp_states.t
        EV_curr = exp_states.EV_curr
        TV_pred = exp_states.TV_pred
        score = exp_states.score
        feas = exp_states.feas
        actual_collision = exp_states.actual_collision
        ref_col = exp_states.ref_col

        if self.state == "Start":
            state_next = "Free-Driving"
        elif self.state == "Free-Driving":
            if self.toEnd(t, EV_curr):
                state_next = "End"
            elif self.toFD(t, EV_curr, TV_pred):
                state_next = "Free-Driving"
            elif self.toSafeCon(score, EV_curr, TV_pred):
                state_next = "Safe-Confidence"
                strategy_next = "Yield"
            elif self.toSafeYield(score, EV_curr, TV_pred):
                strategy_next = "Safe-Yield"
                strategy_next = "Yield"
            elif self.toHOBCA(score, feas, EV_curr, TV_pred):
                state_next = "HOBCA-Unlocked"

                # Max score
                max_idx = np.argmax(score)
                strategy_tmp = self.strategy_names[max_idx]

                if strategy_tmp == "Yield":
                    raise Exception("%s: Yield is triggered when transition to HOBCA unlocked" % self.state)

                strategy_next = strategy_tmp
            else:
                state_next = "Free-Driving"
        elif self.state == "Safe-Confidence":
            if self.toEnd(t, EV_curr):
                state_next = "End"
            elif self.toFD(t, EV_curr, TV_pred):
                state_next = "Free-Driving"
            elif self.toEB(actual_collision):
                state_next = "Emergency-Break"
            elif self.toSafeYield(score, EV_curr, TV_pred):
                state_next = "Safe-Yield"
                strategy_next = "Yield"
            elif self.toHOBCA(score, feas, EV_curr, TV_pred):
                state_next = "HOBCA-Unlocked"

                # Max score
                max_idx = np.argmax(score)
                strategy_tmp = self.strategy_names[max_idx]

                if strategy_tmp == "Yield":
                    raise Exception("%s: Yield is triggered when transition to HOBCA unlocked" % self.state)

                strategy_next = strategy_tmp
            else:
                state_next = "Safe-Confidence"
                strategy_next = "Yield"
        elif self.state == "Safe-Yield":
            if self.toEnd(t, EV_curr):
                state_next = "End"
            elif self.toFD(t, EV_curr, TV_pred):
                state_next = "Free-Driving"
            elif self.toEB(actual_collision):
                state_next = "Emergency-Break"
            elif self.toSafeCon(score, EV_curr, TV_pred):
                state_next = "Safe-Confidence"
                strategy_next = "Yield"
            elif self.toHOBCA(score, feas, EV_curr, TV_pred):
                state_next = "HOBCA-Unlocked"

                # Max score
                max_idx = np.argmax(score)
                strategy_tmp = self.strategy_names[max_idx]

                if strategy_tmp == "Yield":
                    raise Exception("%s: Yield is triggered when transition to HOBCA unlocked" % self.state)

                strategy_next = strategy_tmp
            else:
                state_next = "Safe-Yield"
                strategy_next = "Yield"
        elif self.state == "Safe-Infeasible":
            if self.toEnd(t, EV_curr):
                state_next = "End"
            elif self.toFD(t, EV_curr, TV_pred):
                state_next = "Free-Driving"
            elif self.toEB(actual_collision):
                state_next = "Emergency-Break"
            elif self.toSafeCon(score, EV_curr, TV_pred):
                state_next = "Safe-Confidence"
                strategy_next = "Yield"
            elif self.toSafeYield(score, EV_curr, TV_pred):
                state_next = "Safe-Yield"
                strategy_next = "Yield"
            elif self.toHOBCA(score, feas, EV_curr, TV_pred):
                state_next = "HOBCA-Unlocked"

                # Max score
                max_idx = np.argmax(score)
                strategy_tmp = self.strategy_names[max_idx]

                if strategy_tmp == "Yield":
                    raise Exception("%s: Yield is triggered when transition to HOBCA unlocked" % self.state)

                strategy_next = strategy_tmp
            else:
                state_next = "Safe-Infeasible"
                strategy_next = "Yield"
        elif self.state == "HOBCA-Unlocked":
            if self.toEnd(t, EV_curr):
                state_next = "End"
            elif self.toFD(t, EV_curr, TV_pred):
                state_next = "Free-Driving"
            elif self.toSafeCon(score, EV_curr, TV_pred):
                state_next = "Safe-Confidence"
                strategy_next = "Yield"
            elif self.toSafeYield(score, EV_curr, TV_pred):
                state_next = "Safe-Yield"
                strategy_next = "Yield"
            elif self.toSafeInfeas(feas, EV_curr, TV_pred):
                state_next = "Safe-Infeasible"
                strategy_next = "Yield"
            elif self.toHOBCA_lock(score, ref_col, feas, EV_curr, TV_pred):
                state_next = "HOBCA-Locked"
                strategy_next = self.strategy
            else:
                state_next = "HOBCA-Unlocked"

                # Max score
                max_idx = np.argmax(score)
                strategy_tmp = self.strategy_names[max_idx]

                if strategy_tmp == "Yield":
                    raise Exception("%s: Yield is triggered when transition to HOBCA unlocked" % self.state)

                strategy_next = strategy_tmp
        elif self.state == "HOBCA-Locked":
            if self.toEnd(t, EV_curr):
                state_next = "End"
            elif self.toFD(t, EV_curr, TV_pred):
                state_next = "Free-Driving"
            elif self.toSafeYield(score, EV_curr, TV_pred):
                state_next = "Safe-Yield"
                strategy_next = "Yield"
            elif self.toSafeInfeas(feas, EV_curr, TV_pred):
                state_next = "Safe-Infeasible"
                strategy_next = "Yield"
            else:
                state_next = "HOBCA-Locked"
                strategy_next = self.strategy
        elif self.state == "Emergency-Break":
            if self.toEnd(t, EV_curr):
                state_next = "End"
            elif self.toEB(actual_collision):
                state_next = "Emergency-Break"
            # ============ Below are just for completing the experimentss
            elif self.toFD(t, EV_curr, TV_pred):
                state_next = "Free-Driving"
            elif self.toSafeCon(score, EV_curr, TV_pred):
                state_next = "Safe-Confidence"
                strategy_next = "Yield"
            elif self.toSafeYield(score, EV_curr, TV_pred):
                state_next = "Safe-Yield"
                strategy_next = "Yield"
            elif self.toSafeInfeas(feas, EV_curr, TV_pred):
                state_next = "Safe-Infeasible"
                strategy_next = "Yield"
            elif self.toHOBCA(score, feas, EV_curr, TV_pred):
                state_next = "HOBCA-Unlocked"

                # Max score
                max_idx = np.argmax(score)
                strategy_tmp = self.strategy_names[max_idx]

                if strategy_tmp == "Yield":
                    raise Exception("%s: Yield is triggered when transition to HOBCA unlocked" % self.state)

                strategy_next = strategy_tmp
            else:
                raise Exception("%s: Unexpected Situation" % self.state)
        else:
            raise Exception("%s: Current State is not recoginzed" % self.state)

        self.state = state_next
        self.strategy = strategy_next

        print("The Operation State is now [%s], with strategy [%s]" % (self.state, self.strategy))


    # ============= Transition Conditions
    def toEnd(self, t, EV_curr):
        """
        The transition criteria to End
        """
        if EV_curr[0] >= self.x_max or t >= self.T - self.N:
            return True
        else:
            return False

    def toEB(self, actual_collision):
        """
        The Transition criteria to Emergency Break
        """
        if actual_collision:
            return True
        else:
            return False

    def toFD(self, t, EV_curr, TV_pred):
        """
        The transition to Free Driving
        """
        rel_state = TV_pred - EV_curr

        if np.all( rel_state[0, :] > 20 ) or rel_state[0, 0] < -self.collision_buffer_r or t > self.T_tv - self.N:
            return True
        else:
            return False

    def toSafeCon(self, score, EV_curr, TV_pred):
        """
        The transition criteria to Safety Control - Confidence
        """
        rel_state = TV_pred - TV_pred

        if np.all( rel_state[0, :] > 20 ) or rel_state[0, 0] < -self.collision_buffer_r:
            return False

        # Compute the distance threshold for applying braking assuming max
	    # decceleration is applied

        TV_v = la.norm(TV_pred[3:4, 0])
        TV_th = TV_pred[2, 0]

        EV_v = la.norm(EV_curr[3:4])
        EV_th = EV_curr[2]

        rel_vx = TV_v * np.cos(TV_th) - EV_v * np.cos(EV_th)
        min_ts = np.ceil(-rel_vx / np.abs(self.a_lim[0]) / self.dt) # Number of timesteps requred for relative velocity to be zero
        v_brake = np.abs(rel_vx) + np.arange(min_ts+1) * self.dt * self.a_lim[0] # Velocity when applying max decceleration
        brake_thresh = np.sum( np.abs(v_brake) * self.dt ) + 1 * self.collision_buffer_r # Distance threshold for safety controller to be applied
        d = la.norm(TV_pred[:2,0] - EV_curr[:2]) # Distance between ego and target vehicles

        # Max score
        max_idx = np.argmax(score)

        # If all scores are below the confidence threshold

        if score[max_idx] <= self.confidence_thresh and d <= brake_thresh:
            return True
        else:
            return False

    def toSafeYield(self, score, EV_curr, TV_pred):
        """
        Transition to Safety Control - Yield
        """

        rel_state = TV_pred - EV_curr

        if np.all( rel_state[0, :] > 20 ) or rel_state[0, 0] < -self.collision_buffer_r:
            return False

        # Max score
        max_idx = np.argmax(score)
        strategy_tmp = self.strategy_names[max_idx]

        # If argmax of score is yield

        if strategy_tmp == "Yield":
            return True
        else:
            return False

    def toSafeInfeas(self, feas, EV_curr, TV_pred):
        """
        Trainsition to Safety Control - Infeasible
        """
        rel_state = TV_pred - EV_curr

        if np.all( rel_state[0, :] > 20 ) or rel_state[0, 0] < -self.collision_buffer_r:
            return False

        if not feas:
            return True
        else:
            return False

    def toHOBCA(self, score, feas, EV_curr, TV_pred):
        """
        Transition to HOBCA - Unlocked
        """
        rel_state = TV_pred - EV_curr

        if np.all( rel_state[0, :] > 20 ) or rel_state[0, 0] < -self.collision_buffer_r:
            return False

        # Max score
        max_idx = np.argmax(score)
        strategy_tmp = self.strategy_names[max_idx]

        if score[max_idx] > self.confidence_thresh and (not strategy_tmp == "Yield") and feas:
            return True
        else:
            return False

    def toHOBCA_lock(self, score, ref_col, feas, EV_curr, TV_pred):
        """
        Transition to HOBCA - Locked
        """
        rel_state = TV_pred - EV_curr

        if np.all( rel_state[0, :] > 20 ) or rel_state[0, 0] < -self.collision_buffer_r:
            return False

        # Max score
        max_idx = np.argmax(score)
        strategy_tmp = self.strategy_names[max_idx]

        if (np.sum(ref_col) >= self.lock_steps) and (score[max_idx] > self.confidence_thresh) and (not strategy_tmp == "Yield") and feas:
            return True
        else:
            return False
