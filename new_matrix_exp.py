import torch

def expm(A):    
    """
    Computes the matrix exponential of A using the scaling and squaring 
    algorithm and the Taylor polynomial approximation. Polynomials are
    efficiently evaluated by means of Sastre formulas.
    
    Inputs:
      - A:         input matrix.
    
    Outputs: 
      - E:         exponential of matrix A.
      - m:         approximation polynomial order used.
      - s:         scaling parameter.
      - nProd:     number of matrix products required by the function.
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat PolitÃ¨cnica de ValÃ¨ncia (Spain)
    http://hipersc.blogs.upv.es
    """
    
    # Select m and s values
    m,s,pA,nProd_ms=select_m_s_expm(A)
    
    # Scaling technique
    scaling(pA,s)
    
    # Get polynomial coefficients and evaluate efficiently the polynomial
    # by means of Sastre formulas
    c=coefs_expm_sastre(m)
    fA,nProd_eval=polyvalm_sastre(c,pA)
    
    #Squaring technique
    E,nProd_sq=squaring(fA,s)
    
    #Total number of matrix products
    nProd=nProd_ms+nProd_eval+nProd_sq
    return E

def series(A):    
    """
    Computes the matrix series sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!}. 
    Polynomials are efficiently evaluated by means of Sastre formulas or
    Paterson-Stockmeyer method.
    
    Inputs:
      - A:         input matrix.
    
    Outputs: 
      - E:         result of the matrix series computation.
     
    Group of High Performance Scientific Computation (HiPerSC)
    Universitat PolitÃ¨cnica de ValÃ¨ncia (Spain)
    http://hipersc.blogs.upv.es
    """    
    
    # Select m and s values
    m,s,pA,nProd_ms=select_m_s_series(A)
    # Scaling technique
    scaling(pA,s)
    
    # Get polynomial coefficients and evaluate efficiently the series
    # by means of Paterson-Stockmeyer algoritm or Sastre formulas
    if m<=15:
        c=coefs_series_sastre(m)
        fA,nProd_eval=polyvalm_sastre(c,pA)
    else:
        c=coefs_series_paterson_stockmeyer(m)
        fA,nProd_eval=polyvalm_paterson_stockmeyer(c,pA)    
    
    #Squaring technique
    E,nProd_sq=squaring(fA,s)
    
    #Total number of matrix products
    nProd=nProd_ms+nProd_eval+nProd_sq
    
    return E
   
def select_m_s_expm(A):    
    """
    Determines the polynomial order and the scaling parameter to compute the
    exponential of matrix A by means of Taylor approximation of order m<=15+, 
    without estimations of matrix power norms.
    
    Inputs:
      - A:     input matrix.
  
    Outputs: 
      - m:     approximation polynomial order used.
      - s:     scaling parameter.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd: number of matrix products required by the function.
    """
    
    thetam1=1.414146900027318e-04  # m=1   relative forward error
    theta=[ 3.911041335783245e-03, # m=2   relative forward error
            6.473615365986790e-02, # m=4   relative forward error
            5.089879332517989e-01, # m=8   relative forward error
            2.111282638833816e+00] # m=15+ relative backward error
    c=[     4/3,                   # m=2   
            6/5,                   # m=4
            10/9,                  # m=8
            1.148757271434994]     # m=15+  especial calculation
    ucm2=[  8.000000000000000e-08, # m=2 
            1.440000000000000e-06, # m=4
            4.032000000000000e-03, # m=8
            5.291109128729858e+05] # m=15

    n=A[0].size(-1)
    pA=[]
    a=[torch.math.inf]*17 # vector of ||A^k|| (if a[i]=inf: not calculated)
    s=0
    nProd=0
    
    #a[0]=torch.linalg.matrix_norm(A,ord=1)
    a[0]=torch.norm(A,p=1,dim=-1).max().item()
    
    # Try with m=0
    m=0
    if a[0]==0:
        pA.append(torch.zeros((n,n),device=A.device))
        return m,s,pA,nProd
    
    pA.append(A)
    # Try with m=1
    m=1
    if a[0]<=thetam1:
        return m,s,pA,nProd
    
    pA.append(torch.matmul(A,A))
    # Try with m=2        
    a[1]=torch.norm(pA[1],p=1,dim=-1).max().item()
    nProd=1    
    if a[1]==0: # Nilpotent matrix
        return m,s,pA,nProd    
    m=2
    a[2]=a[0]*a[1]
    a[3]=a[1]**2
    b=max(1,a[0])*ucm2[0]
    if c[0]*a[2]+a[3]<=b:
        return m,s,pA,nProd

    # Try with m=4  
    m=4
    a[4]=a[3]*a[0]
    a[5]=a[3]*a[1]
    b=max(1,a[0])*ucm2[1]
    if c[1]*a[4]+a[5]<=b:
        return m,s,pA,nProd
    
    # Try with m=8  
    m=8
    a[8]=a[4]*a[3]
    a[9]=a[5]*a[3]
    b=max(1,a[0])*ucm2[2]
    if c[2]*a[8]+a[9]<=b:
        return m,s,pA,nProd
    
    # Try with m=15  
    m=15
    a[15]=a[1]**8
    a[16]=a[15]*a[0]
    b=max(1,a[0])*ucm2[3]
    if c[3]*a[15]+a[16]<=b:
        return m,s,pA,nProd
    
    # Compute s for m=15+
    m=15
    alpha_min=max(a[15]**(1/16), a[16]**(1/17))
    t,s=torch.math.frexp(alpha_min/theta[3])
    # Adjust s if norm(A,1)/theta[3] is a power of 2.
    if t==0.5:
        s=s-1
    
    # Test if s can be reduced
    if s>0:
        sred=s-1
        b=max(1,a[0]/2**sred)*ucm2[3]
        if c[3]*a[15]/2**(16*sred)+a[16]/2**(17*sred)<=b:
            # s can be reduced
            s=sred
            
    return m,s,pA,nProd

def select_m_s_series(A,tol=1e-8):    
    """
    Determines the polynomial order and the scaling parameter to compute the
    matrix series sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!} A by means of Taylor 
    approximation of any order (optimal if m<=15+ or Paterson-Stockmeyer if it 
    is not), without estimations of matrix power norms.
    
    Inputs:
      - A:     input matrix.
      - tol:   tolerance. Default value is 1e-8.      
  
    Outputs: 
      - m:     approximation polynomial order used.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.
      - nProd: number of matrix products required by the function.
    """

    theta=[2.449414747911272e-04, # m=1   relative forward error
           6.210608907104720e-03, # m=2   relative forward error
           9.302627708840490e-02, # m=4   relative forward error
           6.720845560952414e-01, # m=8   relative forward error
           2.566720936848977e+00] # m=15+ relative forward error
    c=     1.859216297856016      # m=15+ especial calculation
    ucm2=  7.666123266874392e+06  # m=15+
    
    n=A[0].size(-1)
    pA=[]
    a=[torch.math.inf]*16 # vector of ||A^k|| (if a[i]=inf: not calculated)
    s=0
    nProd=0
    
    #a[0]=torch.linalg.matrix_norm(A,ord=1)
    a[0]=torch.norm(A,p=1,dim=-1).max().item()
    
    # Try with m=0
    m=0
    if a[0]==0:
        pA.append(torch.zeros((n,n),device=A.device))
        return m,s,pA,nProd
    
    pA.append(A)
    # Try with m=1
    m=1
    if a[0]<=theta[0]:
        return m,s,pA,nProd
    
    pA.append(torch.matmul(A,A))
    # Try with m=2        
    #a[1]=torch.linalg.matrix_norm(pA[1],ord=1)
    a[1]=torch.norm(pA[1],p=1,dim=-1).max().item()
    nProd=1    
    if a[1]==0: # Nilpotent matrix
        return m,s,pA,nProd
    m=2
    if (a[0]*a[1])**(1/3)<=theta[1]:
        return m,s,pA,nProd
         
    # Try with m=4  
    m=4
    if (a[0]*a[1]**2)**(1/5)<=theta[2]:
        return m,s,pA,nProd
    
    # Try with m=8  
    m=8
    if (a[0]*a[1]**4)**(1/9)<=theta[3]:
        return m,s,pA,nProd         
         
    # Try with m=15
    m=15
    a16=a[1]**8
    a17=a[0]*a16
    alpha_min=max(a[1]**(1/2),a17**(1/17))
    if alpha_min<=theta[4] or c*a16+a17<=a[0]*ucm2:
        return m,s,pA,nProd
    
    # Try with m>15
    pA.append(torch.matmul(pA[1],A))
    M=[16, 20, 25, 30, 36, 42, 49, 56, 64]    
    i=1;
    while i<=len(M):
        # Try with m
        m=M[i]
        q=torch.math.floor(torch.math.sqrt(m))
        if q>len(pA):
            pA.append(torch.matmul(pA[-1],A))
        if a[1]**(round(m/2))*a[0]/torch.math.factorial(m+2)+a[1]**(round(m/2)+1)/torch.math.factorial(m+3)<=tol:
            return m,s,pA,nProd
        i=i+1
    return m,s,pA,nProd    
        
def scaling(pA,s):    
    """
    Scaling of A powers.
    Inputs:
      - pA: list with the powers of A nedeed. pA[i] contains A^(i+1), 
            for i=0,1,2,3,...,q-1.
      - s:  scaling parameter.
    Outputs:
      - pA: list with the powers of A after scaling.
    """
    
    if s>0:
        q=len(pA)
        for k in range(q):
            pA[k]=pA[k]/(2**((k+1)*s))
   
def squaring(A,s):    
    """
    Squaring of A.
    Inputs:
      - A:     matrix resulting of the approximation polynomial evaluation.
      - s:     scaling parameter.
    
    Outputs:
      - A:     the exponential of matrix A after squaring technique.
      - nProd: number of matrix products required by the function.
    """
    
    nProd=0
    for i in range(s):
        A=torch.matmul(A,A)
        nProd+=1 
    return A,nProd

def coefs_expm_sastre(m):    
    """
    Provides the approximation Taylor polynomial coefficients for the 
    matrix exponential function to be evaluated by means of the Sastre formulas.
    Inputs:
      - m: approximation Taylor polynomial order.
    Outputs:
      - c: vector of m+1 components with the coefficients of the polynomial 
           ordered in descending powers.
    """
    
    if m==0:
        c=[1]
    elif m==1:
        c=[1, 1]
    elif m==2:
        c=[1, 1, 0.5]
    elif m==4:
        c=[1, 1, 0.5, 0.3333333333333333, 0.25]
    elif m==8:
        c=[4.980119205559973e-03, 1.992047682223989e-02, 7.665265321119147e-02, 8.765009801785554e-01, 1.225521150112075e-01, 2.974307204847627, 0.5, 1, 1]
    elif m==15:
        c=[4.018761610201036e-04, 2.945531440279683e-03, -8.709066576837676e-03, 4.017568440673568e-01, 3.230762888122312e-02, 5.768988513026145, 2.338576034271299e-02, 2.381070373870987e-01, 2.224209172496374, -5.792361707073261, -4.130276365929783e-02, 1.040801735231354e+01, -6.331712455883370e+01, 3.484665863364574e-01, 1, 1]
    return c

def coefs_series_paterson_stockmeyer(m):  
    """
    Provides the polynomial coefficients for the matrix series 
    sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!} to be evaluated by means of the 
    Paterson-Stockmeyer formulas.
    Inputs:
      - m: approximation polynomial order.
    Outputs:
      - p: vector of m+1 components with the coefficients of the polynomial 
           ordered from degree zero to degree m, i.e. in ascending powers.
    """    
    p=[1.0000000000000000e+00, 5.0000000000000000e-01, 1.6666666666666666e-01, 4.1666666666666664e-02, 8.3333333333333332e-03, 1.3888888888888889e-03, 1.9841269841269841e-04, 2.4801587301587302e-05, 2.7557319223985893e-06, 2.7557319223985888e-07, 2.5052108385441720e-08, 2.0876756987868100e-09, 1.6059043836821613e-10, 1.1470745597729725e-11, 7.6471637318198164e-13, 4.7794773323873853e-14, 2.8114572543455206e-15, 1.5619206968586225e-16, 8.2206352466243295e-18, 4.1103176233121648e-19, 1.9572941063391263e-20, 8.8967913924505741e-22, 3.8681701706306835e-23, 1.6117375710961184e-24, 6.4469502843844736e-26, 2.4795962632247972e-27, 9.1836898637955460e-29, 3.2798892370698385e-30, 1.1309962886447718e-31, 3.7699876288159061e-33, 1.2161250415535181e-34, 3.8003907548547441e-36, 1.1516335620771951e-37, 3.3871575355211618e-39, 9.6775929586318907e-41, 2.6882202662866367e-42, 7.2654601791530724e-44, 1.9119632050402823e-45, 4.9024697565135435e-47, 1.2256174391283860e-48, 2.9893108271424051e-50, 7.1174067312914405e-52, 1.6552108677421951e-53, 3.7618428812322623e-55, 8.3596508471828045e-57, 1.8173154015614793e-58, 3.8666285139605940e-60, 8.0554760707512382e-62, 1.6439747083165791e-63, 3.2879494166331584e-65, 6.4469596404571737e-67, 1.2397999308571486e-68, 2.3392451525606576e-70, 4.3319354677049218e-72, 7.8762463049180392e-74, 1.4064725544496498e-75, 2.4674957095607889e-77, 4.2543029475186016e-79, 7.2106829618959346e-81, 1.2017804936493225e-82, 1.9701319568021682e-84, 3.1776321883905938e-86, 5.0438606164930067e-88, 7.8810322132703230e-90] 
    p=p[:m+1]
    return p

def coefs_series_sastre(m):
    """
    Provides the polynomial coefficients for the matrix series 
    sum_{k=0}^{infty}\frac{x^{k}}{(k+1)!} to be evaluated by means of the 
    Sastre formulas.
    Inputs:
      - m: approximation polynomial order.
    Outputs:
      - c: vector of m+1 components with the coefficients of the polynomial 
           ordered in descending powers.
    """
    
    if m==0:
        c=[1]
    elif m==1:
        c=[1, 0.5]
    elif m==2:
        c=[1, 0.5, 0.16666666666666666]
    elif m==4:
        c=[1, 0.5, 0.16666666666666666, 0.25, 0.2]
    elif m==8:
        c=[1.660039735186658e-03, 7.470178808339960e-03, 2.709974334395560e-02, 4.500782732024826e-01, 5.880731295195394e-02, 2.034592904871945, 1.666666666666667e-01, 5.000000000000000e-01, 1]
    elif m==15:
        c=[1.999637069327334e-04, 1.494400064180480e-03, -4.364865621286775e-03, 2.294130909981213e-01, 1.868111448270906e-02, 3.650752775284885, 2.113847295209054e-02, 7.709174196270302e-02, 1.005952341579399, 3.134311599789341, -6.535520701550694e-02, 5.711871755550578, -1.928484398466562e+01, 1.116706435878065e-01, 5.000000000000000e-01, 1]
    return c

def polyvalm_paterson_stockmeyer(p,pA):
    """
    Evaluates the polynomial E = p[0]*I + p[1]*A + p[2]*A^2 + ...+ p[m]*A^m 
    efficiently by means of the Paterson-Stockmeyer algorithm.    
    
    Inputs:
      - p:     vector of length m+1 with the polynomial coefficients in 
               ascending powers.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.    
    Outputs: 
      - E:     polynomial evaluation result.
      - nProd: number of matrix products required by the function.  
    """
    
    n=pA[0].size(-1)
    I=torch.eye(pA[0].size(-1),device=pA[0].device)
    m=len(p)-1
    nProd=0
    c=m 
    q=len(pA)
    k=torch.math.ceil(m/q)
    mIdeal=q*k
    q=q-1
    if m==0:
        E=p[0]*I
    else:
        E=torch.zeros((n,n),device=pA[0].device) 
    for j in range(k,0,-1):
        if j==k:
            inic=q-mIdeal+m
        else:
            inic=q-1
        for i in range(inic,-1,-1):
            E=E+p[c]*pA[i]
            c=c-1
        E=E+p[c]*I
        c=c-1;
        if j!=1:
            E=torch.matmul(E,pA[q])
            nProd+=1
    return E,nProd

def polyvalm_sastre(c,pA):    
    """
    Evaluates the polynomial efficiently by means of the Sastre formulas.    
    
    Inputs:
      - m:     polynomial order.  
      - c:     vector of m+1 componentes with the coefficients in descending 
               powers.
      - pA:    list with the powers of A nedeed, such that pA[i] contains
               A^(i+1), for i=0,1,2,3,...,q-1.     
    Outputs: 
      - E:     polynomial evaluation result.
      - nProd: number of matrix products required by the function.  
    """ 
    m=len(c)-1
    E=torch.eye(pA[0].size(-1),device=pA[0].device)
    if m==0:
        nProd=0
    elif m==1:
        E=c[1]*pA[0]+E
        nProd=0
    elif m==2:
        E=c[2]*pA[1]+c[1]*pA[0]+E
        nProd=0
    elif m==4:
        E=torch.matmul(((pA[1]*c[4]+pA[0])*c[3]+E),pA[1]*c[2])+pA[0]*c[1]+E
        nProd=1
    elif m==8:
        y0s=torch.matmul(pA[1],(c[0]*pA[1]+c[1]*pA[0]))
        E=torch.matmul(y0s+c[2]*pA[1]+c[3]*pA[0],y0s+c[4]*pA[1])+c[5]*y0s+c[6]*pA[1]+c[7]*pA[0]+c[8]*E
        nProd=2
    elif m==15:
        y0s=torch.matmul(pA[1],(c[0]*pA[1]+c[1]*pA[0]))
        y1s=torch.matmul(y0s+c[2]*pA[1]+c[3]*pA[0],y0s+c[4]*pA[1])+c[5]*y0s+c[6]*pA[1]
        E=torch.matmul(y1s+c[7]*pA[1]+c[8]*pA[0],y1s+c[9]*y0s+c[10]*pA[0])+c[11]*y1s+c[12]*y0s+c[13]*pA[1]+c[14]*pA[0]+c[15]*E
        nProd=3        
    return E,nProd
                        