def objective_GSR(X,E,A,alpha,beta,gamma):
    obj_val = alpha*variation(X,A)
    +beta*norm(X,'nuc')
    +gamma*norm(hstack(E),1)
    return obj_val  

def variation(X,A):
    return norm(X-A@X,'fro')**2

def D_operator(X,tau):
    U,S,Vh = svd(X,full_matrices=False)
    S = np.diag(S)
    tmp = Theta_operator(S, tau)
    return U@tmp@Vh

def Theta_operator(X,tau):
    add_mask = X>=tau
    sub_mask = X<=-tau
    zero_mask = (X<tau)&(X>-tau)
    X[add_mask] -= tau
    X[sub_mask] += tau
    X[zero_mask] = 0
    return X

def check_term(prev_objval,new_objval,eps):
    if prev_objval < new_objval:
        return False
    return (prev_objval-new_objval)/prev_objval < eps
    
def GSR(T,A,Mask):
    MAX_ITER = 100
    X = T
    W = np.zeros(T.shape)
    E = np.zeros(T.shape)
    C = np.zeros(T.shape)
    Y1 = np.zeros(T.shape)
    Y2 = np.zeros(T.shape)
    Z = np.zeros(T.shape)
    I = np.eye(A.shape[0])
    eps = 1e-6
    gamma_max = norm(hstack(T),1)
    beta_max = norm(T,'nuc')
    alpha_max = variation(T, A)
    ita_max = norm(T,'fro')**2
    alpha = 1/alpha_max
    beta = 0.18/beta_max
    gamma = 0.3/gamma_max
    ita = 1500/ita_max
    obj_vals = []
    prev_objval = objective_GSR(X,E,A,alpha,beta,gamma)
    inv_term = inv(I+2*alpha*(1/ita)*(I-A)@(I-A))    

    # iteration
    for i in range(MAX_ITER):
        # update X
        # backtracking line search to determine t
        t = 1 # initial value
        c = 0.5# search control parameter
        prev_X_obj = beta*norm(X,'nuc')+(ita/2)*norm(T-X-W-E-C-Y1,'fro')**2+(ita/2)*norm(X-Z-Y2,'fro')**2
        for j in range(MAX_ITER):
            new_X = D_operator(X+t*(T-W-E-C-(1/ita)*(Y1+Y2)-Z), beta*(1/ita))
            new_X_obj = beta*norm(new_X,'nuc')+(ita/2)*norm(T-new_X-W-E-C-Y1,'fro')**2+(ita/2)*norm(new_X-Z-Y2,'fro')**2
            if j!=0 and check_term(prev_X_obj, new_X_obj, eps):
                break
            t = -c*t
            prev_X_obj = new_X_obj
        prev_X_obj = beta*norm(X,'nuc')+(ita/2)*norm(T-X-W-E-C-Y1,'fro')**2+(ita/2)*norm(X-Z-Y2,'fro')**2
        X = new_X
        
        # update W
        W = ita*(T-X-E-C-(1/ita)*Y1)/(ita+2)
        
        # update E
        # backtracking line search to determine t
        t = 1 # initial value
        c = 0.5# search control parameter
        prev_E_obj = gamma*norm(hstack(E),1) + (ita/2)*norm(T-X-W-E-C-Y1,'fro')**2
        for j in range(MAX_ITER):
            new_E = Theta_operator(X+t*(T-X-W-C-(1/ita)*Y1), gamma*(1/ita))
            new_E_obj = gamma*norm(hstack(new_E),1) + (ita/2)*norm(T-X-W-new_E-C-Y1,'fro')**2
            if j!=0 and check_term(prev_E_obj, new_E_obj, eps):
                break
            t = -c*t
            prev_E_obj = new_E_obj
        prev_E_obj = gamma*norm(hstack(E),1) + (ita/2)*norm(T-X-W-E-C-Y1,'fro')**2
        E = new_E
                
        # update parameters
        Z = inv_term@(X-(1/ita)*Y2)
        C[~Mask] = (T-X-W-E-(1/ita)*Y1)[~Mask]
        C[Mask] = 0
        Y1 = Y1 - ita*(T-X-W-E-C)
        Y2 = Y2 - ita*(X-Z)

        # check terminate condition:
        obj_val = objective_GSR(X, E, A, alpha, beta, gamma)
        obj_vals.append(obj_val)
        if check_term(prev_objval, obj_val, eps):
            break
        prev_objval = obj_val
    return X,W,E,obj_vals
