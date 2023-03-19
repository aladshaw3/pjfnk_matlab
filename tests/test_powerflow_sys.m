% Integration tests for power flow problems

%% Test 1 - Use Newton Method for Y-bus matrix (Y*V = I)

%         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
%         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata = [1      2       0.01938   0.05917    0.0264         1
             1      5       0.05403   0.22304    0.0246         1
             2      3       0.04699   0.19797    0.0219         1
             2      4       0.05811   0.17632    0.0170         1
             2      5       0.05695   0.17388    0.0173         1
             3      4       0.06701   0.17103    0.0064         1
             4      5       0.01335   0.04211    0.0            1
             4      7       0.0       0.20912    0.0        0.978
             4      9       0.0       0.55618    0.0        0.969
             5      6       0.0       0.25202    0.0        0.932
             6     11       0.09498   0.19890    0.0            1
             6     12       0.12291   0.25581    0.0            1
             6     13       0.06615   0.13027    0.0            1
             7      8       0.0       0.17615    0.0            1
             7      9       0.0       0.11001    0.0            1
             9     10       0.03181   0.08450    0.0            1
             9     14       0.12711   0.27038    0.0            1
            10     11       0.08205   0.19207    0.0            1
            12     13       0.22092   0.19988    0.0            1
            13     14       0.17093   0.34802    0.0            1 ];

Y = YbusFormation(linedata);
rng(211);
I = rand(size(Y,1),1);
V0 = zeros(size(Y,1),1);

F = @(V) [Y*V-I];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = Y;

tic;
[V,stats,opts] = NewtonMethod(F,V0,solver_info);
toc;

assert( stats.fnorm(end) < 1e-6 )

%% Test 2 - JFNK using PCG for different Y-bus system

% NOTE: PCG requires the system to be symmetric-positive-definite (SPD), 
% but in general this won't be true of all Y-bus matrices, which are always
% symmetric, but not always positive definite (i.e., they don't always have
% all positive eigenvalues)

% For a power flow system of all simple resistors, we can make ths system
% to be symmetric and positive definite, for the purpose of this test

%         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
%         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata = [1      2       0.01938   0.0    0.0         1
             1      5       0.05403   0.0    0.0        1
             2      3       0.04699   0.0    0.0        1
             2      4       0.05811   0.0    0.0        1
             2      5       0.05695   0.0    0.0        1
             3      4       0.06701   0.0    0.0        1
             4      5       0.01335   0.0    0.0            1
             4      7       0.5       0.0   0.0        0.978
             4      9       0.5       0.0    0.0        0.969
             5      6       0.5       0.0    0.0        0.932
             6     11       0.09498   0.0    0.0            1
             6     12       0.12291   0.0    0.0            1
             6     13       0.06615   0.0    0.0            1
             7      8       0.5       0.0    0.0            1
             7      9       0.5      0.0    0.0            1
             9     10       0.03181   0.0    0.0            1
             9     14       0.12711   0.0    0.0            1
            10     11       0.08205   0.0    0.0            1
            12     13       0.22092   0.0    0.0            1
            13     14       0.17093   0.0    0.0            1 ];

Y = YbusFormation(linedata);
rng(211);
I = ones(size(Y,1),1)*1e-3;
V0 = zeros(size(Y,1),1);

F = @(V) [Y*V-I];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = Y;
solver_info.krylov_solver = 'pcg';
solver_info.krylov_opts = struct();

% Create custom preconditioner
solver_info.krylov_opts.user_data = struct('type','ichol');
solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) precon(b,Jacfun,x,options);

tic;
[V,stats,opts] = JacobianFreeNewtonKrylov(F,V0,solver_info);
toc;

assert( stats.fnorm(end) < 1e-6 )


%% Test 3 - JFNK with Minres Krylov method

% MINRES requires the system to be symmetric, but doesn't need to the SPD.

%         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
%         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata = [1      2       0.01938   0.05917    0.0264         1
             1      5       0.05403   0.22304    0.0246         1
             2      3       0.04699   0.19797    0.0219         1
             2      4       0.05811   0.17632    0.0170         1
             2      5       0.05695   0.17388    0.0173         1
             3      4       0.06701   0.17103    0.0064         1
             4      5       0.01335   0.04211    0.0            1
             4      7       0.0       0.20912    0.0        0.978
             4      9       0.0       0.55618    0.0        0.969
             5      6       0.0       0.25202    0.0        0.932
             6     11       0.09498   0.19890    0.0            1
             6     12       0.12291   0.25581    0.0            1
             6     13       0.06615   0.13027    0.0            1
             7      8       0.0       0.17615    0.0            1
             7      9       0.0       0.11001    0.0            1
             9     10       0.03181   0.08450    0.0            1
             9     14       0.12711   0.27038    0.0            1
            10     11       0.08205   0.19207    0.0            1
            12     13       0.22092   0.19988    0.0            1
            13     14       0.17093   0.34802    0.0            1 ];

Y = YbusFormation(linedata);
rng(200);
I = rand(size(Y,1),1);
V0 = zeros(size(Y,1),1);

F = @(V) [Y*V-I];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = Y;
solver_info.krylov_solver = 'minres';
solver_info.krylov_opts = struct();

% Create custom preconditioner
solver_info.krylov_opts.user_data = struct('type','ilu');
solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) precon(b,Jacfun,x,options);

tic;
[V,stats,opts] = JacobianFreeNewtonKrylov(F,V0,solver_info);
toc;

assert( stats.fnorm(end) < 1e-6 )


%% Test 4 - JFNK with SymmLQ Krylov method

% SymmLQ requires the system to be symmetric, but doesn't need to the SPD.

%         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
%         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata = [1      2       0.01938   0.05917    0.0264         1
             1      5       0.05403   0.22304    0.0246         1
             2      3       0.04699   0.19797    0.0219         1
             2      4       0.05811   0.17632    0.0170         1
             2      5       0.05695   0.17388    0.0173         1
             3      4       0.06701   0.17103    0.0064         1
             4      5       0.01335   0.04211    0.0            1
             4      7       0.0       0.20912    0.0        0.978
             4      9       0.0       0.55618    0.0        0.969
             5      6       0.0       0.25202    0.0        0.932
             6     11       0.09498   0.19890    0.0            1
             6     12       0.12291   0.25581    0.0            1
             6     13       0.06615   0.13027    0.0            1
             7      8       0.0       0.17615    0.0            1
             7      9       0.0       0.11001    0.0            1
             9     10       0.03181   0.08450    0.0            1
             9     14       0.12711   0.27038    0.0            1
            10     11       0.08205   0.19207    0.0            1
            12     13       0.22092   0.19988    0.0            1
            13     14       0.17093   0.34802    0.0            1 ];

Y = YbusFormation(linedata);
rng(190);
I = rand(size(Y,1),1);
V0 = zeros(size(Y,1),1);

F = @(V) [Y*V-I];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = Y;
solver_info.krylov_solver = 'symmlq';
solver_info.use_matrix = true;
solver_info.krylov_opts = struct();

% Create custom preconditioner
%solver_info.krylov_opts.user_data = struct('type','ilu');
%solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) precon(b,Jacfun,x,options);

tic;
[V,stats,opts] = JacobianFreeNewtonKrylov(F,V0,solver_info);
toc;

assert( stats.fnorm(end) < 1e-6 )


%% Test 5 - JFNK with BiCGStab(l) Krylov method

% BiCGStab(l) requires the system to be square. 

%         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
%         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata = [1      2       0.01938   0.05917    0.0264         1
             1      5       0.05403   0.22304    0.0246         1
             2      3       0.04699   0.19797    0.0219         1
             2      4       0.05811   0.17632    0.0170         1
             2      5       0.05695   0.17388    0.0173         1
             3      4       0.06701   0.17103    0.0064         1
             4      5       0.01335   0.04211    0.0            1
             4      7       0.0       0.20912    0.0        0.978
             4      9       0.0       0.55618    0.0        0.969
             5      6       0.0       0.25202    0.0        0.932
             6     11       0.09498   0.19890    0.0            1
             6     12       0.12291   0.25581    0.0            1
             6     13       0.06615   0.13027    0.0            1
             7      8       0.0       0.17615    0.0            1
             7      9       0.0       0.11001    0.0            1
             9     10       0.03181   0.08450    0.0            1
             9     14       0.12711   0.27038    0.0            1
            10     11       0.08205   0.19207    0.0            1
            12     13       0.22092   0.19988    0.0            1
            13     14       0.17093   0.34802    0.0            1 ];

Y = YbusFormation(linedata);
rng(180);
I = rand(size(Y,1),1);
V0 = zeros(size(Y,1),1);

F = @(V) [Y*V-I];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = Y;
solver_info.krylov_solver = 'bicgstabl';
solver_info.use_matrix = true;
solver_info.krylov_opts = struct();

% Create custom preconditioner
%solver_info.krylov_opts.user_data = struct('type','ilu');
%solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) precon(b,Jacfun,x,options);

tic;
[V,stats,opts] = JacobianFreeNewtonKrylov(F,V0,solver_info);
toc;

assert( stats.fnorm(end) < 1e-6 )


%% Test 6 - JFNK with CGS Krylov method

% CGS requires the system to be square. Similar to BiCG, but without
% needing the transpose function. 

%         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
%         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata = [1      2       0.01938   0.05917    0.0264         1
             1      5       0.05403   0.22304    0.0246         1
             2      3       0.04699   0.19797    0.0219         1
             2      4       0.05811   0.17632    0.0170         1
             2      5       0.05695   0.17388    0.0173         1
             3      4       0.06701   0.17103    0.0064         1
             4      5       0.01335   0.04211    0.0            1
             4      7       0.0       0.20912    0.0        0.978
             4      9       0.0       0.55618    0.0        0.969
             5      6       0.0       0.25202    0.0        0.932
             6     11       0.09498   0.19890    0.0            1
             6     12       0.12291   0.25581    0.0            1
             6     13       0.06615   0.13027    0.0            1
             7      8       0.0       0.17615    0.0            1
             7      9       0.0       0.11001    0.0            1
             9     10       0.03181   0.08450    0.0            1
             9     14       0.12711   0.27038    0.0            1
            10     11       0.08205   0.19207    0.0            1
            12     13       0.22092   0.19988    0.0            1
            13     14       0.17093   0.34802    0.0            1 ];

Y = YbusFormation(linedata);
rng(150);
I = rand(size(Y,1),1);
V0 = zeros(size(Y,1),1);

F = @(V) [Y*V-I];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = Y;
solver_info.krylov_solver = 'cgs';
solver_info.use_matrix = true;
solver_info.krylov_opts = struct();

% Create custom preconditioner
solver_info.krylov_opts.user_data = struct('type','ilu');
solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) precon(b,Jacfun,x,options);

tic;
[V,stats,opts] = JacobianFreeNewtonKrylov(F,V0,solver_info);
toc;

assert( stats.fnorm(end) < 1e-6 )


%% Test 7 - JFNK with TFQMR Krylov method

% TFQMR requires the system to be square. Generally the best solver for
% symmetric problems that are 'indefinite' (most relevant for Power Flow)

%         |  From |  To   |   R     |   X     |     B/2  |  X'mer  |
%         |  Bus  | Bus   |  pu     |  pu     |     pu   | TAP (a) |
linedata = [   1    2    0.0083    0.0280      0.0645         1 
                2    3    0.0298    0.0850      0.0409         1
                3    4    0.0112    0.0366      0.0190         1
                4    5    0.0625    0.1320      0.0129         1
                4    6    0.0430    0.1480      0.0174         1
                6    7    0.0200    0.1020      0.0138         1
                6    8    0.0339    0.1730      0.0235         1
                8    9    0.0099    0.0505      0.0274         1
                9   10    0.0369    0.1679      0.0220         1
                9   11    0.0258    0.0848      0.0109         1
                9   12    0.0648    0.2950      0.0386         1
                9   13    0.0481    0.1580      0.0203         1
               13   14    0.0132    0.0434      0.0055         1
               13   15    0.0269    0.0869      0.0115         1
                1   15    0.0178    0.0910      0.0494         1
                1   16    0.0454    0.2060      0.0273         1
                1   17    0.0238    0.1080      0.0143         1
                3   15    0.0162    0.0530      0.0272         1
                4   18    0.0       0.5550      0.0        0.970
                4   18    0.0       0.4300      0.0        0.978
                5    6    0.0302    0.0641      0.0062         1
                7    8    0.0139    0.0712      0.0097         1
               10   12    0.0277    0.1262      0.0164         1
               11   13    0.0223    0.0732      0.0094         1
               12   13    0.0178    0.0580      0.0302         1
               12   16    0.0180    0.0813      0.0108         1
               12   17    0.0397    0.1790      0.0238         1
               14   15    0.0171    0.0547      0.0074         1
               18   19    0.4610    0.6850      0.0            1
               19   20    0.2830    0.4340      0.0            1
               21   20    0.0       0.7767      0.0        1.043
               21   22    0.0736    0.1170      0.0            1
               22   23    0.0099    0.0152      0.0            1
               23   24    0.1660    0.2560      0.0042         1
               24   25    0.0       1.1820      0.0            1    
               24   25    0.0       1.2300      0.0            1    
               24   26    0.0       0.0473      0.0        1.043
               26   27    0.1650    0.2540      0.0            1
               27   28    0.0618    0.0954      0.0            1
               28   29    0.0418    0.0587      0.0            1
                7   29    0.0       0.0648      0.0        0.967
               25   30    0.1350    0.2020      0.0            1
               30   31    0.3260    0.4970      0.0            1
               31   32    0.5070    0.7550      0.0            1
               32   33    0.0392    0.0360      0.0            1
               34   32    0.0       0.9530      0.0        0.975
               34   35    0.0520    0.0780      0.0016         1
               35   36    0.0430    0.0537      0.0008         1
               36   37    0.0290    0.0366      0.0            1
               37   38    0.0651    0.1009      0.0010         1
               37   39    0.0239    0.0379      0.0            1
               36   40    0.0300    0.0466      0.0            1
               22   38    0.0192    0.0295      0.0            1
               11   41    0.0       0.7490      0.0        0.955
               41   42    0.2070    0.3520      0.0            1
               41   43    0.0       0.4120      0.0            1
               38   44    0.0289    0.0585      0.0010         1
               15   45    0.0       0.1042      0.0        0.955
               14   46    0.0       0.0735      0.0        0.900
               46   47    0.0230    0.0680      0.0016         1
               47   48    0.0182    0.0233      0.0            1
               48   49    0.0834    0.1290      0.0024         1
               49   50    0.0801    0.1280      0.0            1
               50   51    0.1386    0.2200      0.0            1
               10   51    0.0       0.0712      0.0        0.930
               13   49    0.0       0.1910      0.0        0.895
               29   52    0.1442    0.1870      0.0            1
               52   53    0.0762    0.0984      0.0            1
               53   54    0.1878    0.2320      0.0            1
               54   55    0.1732    0.2265      0.0            1
               11   43    0.0       0.1530      0.0        0.958
               44   45    0.0624    0.1242      0.0020         1
               40   56    0.0       1.1950      0.0        0.958
               56   41    0.5530    0.5490      0.0            1
               56   42    0.2125    0.3540      0.0            1
               39   57    0.0       1.3550      0.0        0.980
               57   56    0.1740    0.2600      0.0            1
               38   49    0.1150    0.1770      0.0015         1
               38   48    0.0312    0.0482      0.0            1
                9   55    0.0       0.1205      0.0        0.940];

Y = YbusFormation(linedata);
rng(150);
I = rand(size(Y,1),1);
V0 = zeros(size(Y,1),1);

F = @(V) [Y*V-I];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = Y;
solver_info.krylov_solver = 'tfqmr';
solver_info.use_matrix = false;
solver_info.krylov_opts = struct();

% Create custom preconditioner
solver_info.krylov_opts.user_data = struct('type','ilu');
solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) precon(b,Jacfun,x,options);

tic;
[V,stats,opts] = JacobianFreeNewtonKrylov(F,V0,solver_info);
toc;

assert( stats.fnorm(end) < 1e-6 )


%% Helper function to return Y-bus matrix
function Y = YbusFormation(linedata)
    fb = linedata(:,1);     % From bus number...
    tb = linedata(:,2);     % To bus number...
    r = linedata(:,3);      % Resistance, R...
    x = linedata(:,4);      % Reactance, X...
    b = linedata(:,5);      % Ground Admittance, B/2...
    a = linedata(:,6);      % Tap setting value..
    z = r + 1i*x;            % Z matrix...
    y = 1./z;               % To get inverse of each element...
    b = 1i*b;                % Make B imaginary...
    nbus = max(max(fb),max(tb));    % no. of buses...
    nbranch = length(fb);           % no. of branches...
    Y = zeros(nbus,nbus);        % Initialise YBus...
     
    % Formation of the Off Diagonal Elements...
    for k=1:nbranch
     Y(fb(k),tb(k)) = Y(fb(k),tb(k))-y(k)/a(k);
     Y(tb(k),fb(k)) = Y(fb(k),tb(k));
    end
    
    % Formation of Diagonal Elements....
    for m =1:nbus
        for n =1:nbranch
           if fb(n) == m
               Y(m,m) = Y(m,m) + y(n)/(a(n)^2) + b(n);
           elseif tb(n) == m
               Y(m,m) = Y(m,m) + y(n) + b(n);
           end
        end
    end
end

%% Helper function for ilu and ichol preconditioners
% Defined preconditioner helper
%       NOTE: x comes in as 'b' (or '-F') in this context
function y = precon(b,Jac,x,options)
    if (strcmp(options.type,'ilu'))
        [L,U] = ilu(Jac(x),struct('type','ilutp','droptol',1e-6));
        y = U\(L\b);
    else
        L = ichol(Jac(x),struct('type','nofill','diagcomp',1e-6));
        y = L'\(L\b);
    end
end