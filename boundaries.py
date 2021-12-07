def Define_boundaries(num_function, xu, xl, n, m) :
    if num_function == 1 :
        o  = 1; # number of objectives 
        n  = 6; # number of variables 
        ni = 0; # number of integer variables 
        m  = 4; # number of constraints    
        xl = [-1000.0,30.0,100.0,30.0,400.0,1000.0] # lower bounds
        xu = [0.0,400.0,470.0,400.0,2000.0,6000.0]  # upper bounds

    elif num_function == 2 :
        o  = 1; # number of objectives 
        n  = 22; # number of variables 
        ni = 0; # number of integer variables 
        m  = 0; # number of constraints    
        xl = [-1000.0,3.0,0.0,0.0,100.0,100.0,30.0,400.0,800.0,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.15,1.7,-pi,-pi,-pi,-pi]
        xu = [0.0,5.0,1.0,1.0,400.0,500.0,300.0,1600.0,2200.0,0.9,0.9,0.9,0.9,0.9,6.0,6.0,6.5,291.0,pi,pi,pi,pi]

    elif num_function == 3 :
        o  = 1; # number of objectives 
        n  = 18; # number of variables 
        ni = 0; # number of integer variables 
        m  = 0; # number of constraints    
        xl = [1000.0,1.0,0.0,0.0,30.0,30.0,30.0,30.0,0.01,0.01,0.01,0.01,1.1,1.1,1.1,-pi,-pi,-pi]
        xu = [4000.0,5.0,1.0,1.0,400.0,400.0,400.0,400.0,0.99,0.99,0.99,0.99,6.0,6.0,6.0,pi,pi,pi]

    elif num_function == 4 :
        o  = 1; # number of objectives 
        n  = 26; # number of variables 
        ni = 0; # number of integer variables 
        m  = 0; # number of constraints    
        xl = [1900.0,2.5,0.0,0.0,100.0,100.0,100.0,100.0,100.0,100.0,0.01,0.01,0.01,0.01,0.01,0.01,1.1,1.1,1.05,1.05,1.05,  -pi,  -pi,  -pi,  -pi,  -pi]
        xu = [2300.0,4.05,1.0,1.0,500.0,500.0,500.0,500.0,500.0,600.0,0.99,0.99,0.99,0.99,0.99,0.99,6.0,6.0,6.0,6.0,6.0,   pi,  pi,  pi,  pi,  pi]

    elif num_function == 5 :
        o  = 1; # number of objectives 
        n  = 8; # number of variables 
        ni = 0; # number of integer variables 
        m  = 6; # number of constraints    
        xl = [3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0]
        xu = [10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0]

    elif num_function == 6 :
        o  = 1; # number of objectives 
        n  = 22; # number of variables 
        ni = 0; # number of integer variables 
        m  = 0; # number of constraints    
        xl = [1460.0,3.0,0.0,0.0,300.0,150.0,150.0,300.0,700.0,0.01,0.01,0.01,0.01,0.01,1.06,1.05,1.05,1.05,-pi,-pi,-pi,-pi]
        xu = [1825.0,5.0,1.0,1.0,500.0,800.0,800.0,800.0,1850.0,0.9,0.9,0.9,0.9,0.9,9.0,9.0,9.0,9.0,pi,pi,pi,pi]

    elif num_function == 7 :
        o  = 1; # number of objectives 
        n  = 12; # number of variables 
        ni = 0; # number of integer variables 
        m  = 2; # number of constraints    
        xl = [7000.0, 0.0, 0.0, 0.0,   50.0,  300.0, 0.01, 0.01, 1.05,   8.0, -pi, -pi]
        xu = [9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0,  0.9,  0.9,    7.0, 500.0,  pi,  pi]

    elif num_function == 8 :
        o  = 1; # number of objectives 
        n  = 10; # number of variables 
        ni = 4; # number of integer variables 
        m  = 4; # number of constraints    
        xl = [-1000.0,30.0,100.0,30.0,400.0,1000.0,   1.0,1.0,1.0,1.0]
        xu = [0.0,400.0,470.0,400.0,2000.0,6000.0,    9.0,9.0,9.0,9.0]