#Dependences
import numpy as np

#Custom adaptation of the Kalman filter - which is not a Kalman filter at all
#       -> Replaces Kalman's inovation with a NN
#Requirements for targets
#   -> state = [xc, yc, w, h, x', y', w', h']
class NNFilter(object):
    def __init__(self, screen_w, screen_h, weight_path):
        #CNN params
        self.coeff_model = self.FCNN(weight_path)

        self.d = 1
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.S = screen_w / 2
        self.grad = []
        self.incr = 1
        self.pre_calc(self.incr)

    #--------------------------Motion fucntions--------------------------
    #Calculate gradient at pixel point px,py : returns [dpx/dx, dpx/dy, dpy/dx, dpy/dy] / S where S in the radius of the fisheye image in pixels
    def calc_gradient(self, px, py): #TODO Pray that this is correct
        #return [1,0,1,0]
        px = px - self.screen_w/2
        py = py - self.screen_h/2
        normpxpy = np.sqrt(px*px + py*py)
        
        if (normpxpy < 1e-10):
            return [1, 1, 1, 1] #TODO Maybe choose better values
        elif (normpxpy > self.screen_w/2):
            return [0, 0, 0, 0]

        normxy = self.d*np.tan(np.arcsin(2*normpxpy/self.screen_w))
        x = px*normxy/normpxpy
        y = np.sqrt(max(0, normxy**2 - x**2)) * np.sign(py)

        n2 = normxy**2
        n3 = normxy**3

        A1 = np.arctan(normxy/self.d)
        A = np.sin(A1)
        A2 = np.cos(A1)/(1 + n2)
        dAdx = A2*x/(normxy*self.d)
        dAdy = A2*y/(normxy*self.d)

        B = x/normxy
        dBdx = 1/normxy - B*x
        dBdy = -1*x*y/(n3)

        C = np.sin(np.arccos(max(-1, min(1, x/normxy))))
        C1 = 0
        if (y > 1e-10):
            C1 = x/(np.sqrt(n2 - x**2))
        dCdx = (x**2/n3 - 1/normxy)*C1
        dCdy = x*y/n3*C1

        return [dAdx*B + A*dBdx, dAdy*B + A*dBdy, dAdx*C + A*dCdx, dAdy*C + A*dCdy]
    
    #Calculate matrix of gradient - option save -> save it at path save_path
    def pre_calc(self, incr, save_path = "", d = -1):
        if d > 0:
            self.d = d
        
        self.incr = incr

        dim = (int(self.screen_w/incr), int(self.screen_h/incr), 4)
        arr = np.zeros(dim)
        for x in range(dim[0]):
            for y in range(dim[1]):
                arr[x, y] = np.array(self.calc_gradient(x*incr, y*incr))
        
        #Save?
        if save_path != "":
            np.save(save_path, arr, allow_pickle = True)
        
        self.grad = arr #* self.S #TODO why the fuck does it work better without self.S
    
    #Load pre_calc matrix
    def load_pre_calc(self, save_path):
        self.grad = np.load(save_path, allow_pickle = True) #* self.S
        self.incr = self.screen_w / self.grad.shape[0]
    
    #Function to get gradient
    #   - Values are clamped btw 0 and max
    def get_gradient(self, state):
        pc = (np.clip(int(state[0] / self.incr), 0, self.screen_w - 1), 
        np.clip(int(state[1] / self.incr), 0, self.screen_h - 1))
        p1 = (np.clip(int((state[0] - state[2]/2) / self.incr), 0, self.screen_w - 1), 
        np.clip(int((state[1] - state[3]/2) / self.incr), 0, self.screen_h - 1))
        p2 = (np.clip(int((state[0] + state[2]/2) / self.incr), 0, self.screen_w - 1), 
        np.clip(int((state[1] + state[3]/2) / self.incr), 0, self.screen_h - 1))
        return [self.grad[pc[0], pc[1]], self.grad[p2[0], p2[1]] - self.grad[p1[0], p1[1]]]


    #Get motion matrix with gradients already in place TODO Does this shit work?
    def get_motion_matrix(self, state, targ, dt):
        grad = self.get_gradient(state)
        targ.last_grad = grad
        return np.array([
            [1, 0, 0, 0, dt * grad[0][0], dt * grad[0][1], 0, 0],
            [0, 1, 0, 0, dt * grad[0][2], dt * grad[0][3], 0, 0],
            [0, 0, 1, 0, dt * grad[1][0], dt * grad[1][1], dt, 0],
            [0, 0, 0, 1, dt * grad[1][2], dt * grad[1][3], 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]])

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
        targ.pred_state = np.matmul(self.get_motion_matrix(targ.state, targ, dt), targ.state)
    
    #   State update - no detection
    def update_state_no_detection(self, targ, dt):
        targ.state = targ.pred_state

        targ.missed_detection = True
        targ.time_since_last_detection += dt
        targ.age += 1

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
        a1 = (targ.last_grad[0][0]*targ.last_grad[0][3] - targ.last_grad[0][1]*targ.last_grad[0][2]) * ndt
        if(a1 < 1e-10): #TODO WTF is happening here...
            a1 = 1
        else:
            a1 = 1 / a1
        vx = (targ.last_grad[0][3] * dpx - targ.last_grad[0][1] * dpy) * a1
        vy = (targ.last_grad[0][0] * dpy - targ.last_grad[0][2] * dpx) * a1
        vw = dpw / ndt - (targ.last_grad[1][0] * vx + targ.last_grad[1][1] * vy)
        vh = dph / ndt - (targ.last_grad[1][2] * vx + targ.last_grad[1][3] * vy)
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