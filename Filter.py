#Dependences
import numpy as np

from Target import target

#Custom adaptation of the Kalman filter - which is not a Kalman filter at all
#       -> Replaces Kalman's inovation with a NN
#Requirements for targets
#   -> state = [xc, yc, w, h, x', y', w', h']
class NNFilter(object):
    def __init__(self, weight_path):# screen_w, screen_h, weight_path):
        #CNN params
        self.coeff_model = self.FCNN(weight_path)
        # self.screen_w = screen_w
        # self.screen_h = screen_h

    #--------------------------NN functions--------------------------
    #   NN class
    #       -> The network is fully connected and small => nothing complicated
    #       -> Keras has to many things slowing down calculations for such a small network
    #       -> So we make a custom feedforward class
    #       -> Obligatory sigmoid because I am lazy
    class FCNN(object):
        def __init__(self, weight_path):
            if(weight_path == ""):
                return None
            all = np.load(weight_path, allow_pickle = True)
            self.weights = all[0]
            self.bias = all[1]
            self.n = len(self.weights)
        
        def sigmoid(self, inp):
            exp = np.exp(inp)
            return exp / (1 + exp)
        
        def predict(self, inp):
            for i in range(self.n):
                inp = np.matmul(inp, self.weights[i]) + self.bias[i]
                inp = self.sigmoid(inp)
            return inp

    #--------------------------State functions--------------------------
    #   State prediction
    def pred_next_state(self, targ, dt):
        targ.pred_state = targ.state
        targ.pred_state[:4] += targ.state[4:] * dt
    
    #   State update - no detection
    def update_state_no_detection(self, targ, dt):
        targ.state = targ.pred_state

        targ.missed_detection = True
        targ.time_since_last_detection += dt
        targ.age += 1

     #TODO : Testing
    # def update_state(self, targ, detect, dt):
    #     def get_length(v):
    #         return v[0]**2 + v[1]**2
    #     dist = get_length(detect[:2] - targ.pred_state.T[0][:2])
    #     if dist > 0.4:
    #         return target(detect, targ.features, targ.id)
    #     return self.update_state_bis(targ, detect, dt)

    #   State update - detection (detect -> [xc, yc, w, h])
    def update_state(self, targ, detect, dt):
        #Calculate error vector
        #   Calculate detection
        dpx, dpy, dpw, dph, ndt = 0, 0, 0, 0, 0
        if(targ.missed_detection): # TODO Make sure this is fine - here dt could be big and fuck up the taylor dev (Update : Taylor gone rn)
            dpx = detect[0] - targ.last_detected_state[0,0]
            dpy = detect[1] - targ.last_detected_state[1,0]
            dpw = detect[2] - targ.last_detected_state[2,0]
            dph = detect[3] - targ.last_detected_state[3,0]
            ndt = targ.time_since_last_detection
            targ.missed_detection = False
            targ.time_since_last_detection = 0
        else:
            dpx = detect[0] - targ.state[0,0]
            dpy = detect[1] - targ.state[1,0]
            dpw = detect[2] - targ.state[2,0]
            dph = detect[3] - targ.state[3,0]
            ndt = dt
        vx = dpx / ndt
        vy = dpy / ndt
        vw = dpw / ndt
        vh = dph / ndt

        # if(vx*vx + vy*vy > 10):
        #     targ.time_since_last_detection = ndt
        #     self.update_state_no_detection(targ, dt)
        #     return np.array([[]]), np.array([[]])

        detected_state = np.array([[detect[0], detect[1], detect[2], detect[3], vx, vy, vw, vh]]).T
    	#   Calculate error
        err = detected_state - targ.pred_state

        #Calculate interpolation vector
        # inp = np.array([[err[0][0] / self.screen_w, err[1][0] / self.screen_h, 0 if targ.missed_detection else 1]])
        inp = np.array([[err[0][0], err[1][0], detect[0] - 0.5, detect[1] - 0.5, 0 if targ.missed_detection else 1]])
        coeff = self.coeff_model.predict(inp)

        #Update state
        targ.state = targ.pred_state + coeff.T * err
        targ.last_detected_state = targ.state
        targ.age = 0

        return err, inp