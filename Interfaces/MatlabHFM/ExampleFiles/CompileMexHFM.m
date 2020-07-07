% Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay

if verLessThan('matlab','8.1')
    cxxFlags = ['CXXFLAGS="-std=c++17 -fPIC" ' ...
        'CXXLIBS="\$CXXLIBS -lc++" ' ]; % This flag is required on some platforms, but must be commented on others...
    outputFlag = '-o ';
else
    cxxFlags = 'CXXFLAGS="-std=c++17 -fPIC" ';
    outputFlag = '-output ';
end

compileHFM = @(binary_Dir,name,flag) eval(['mex ' ...
    outputFlag 'MatlabHFM_' name ' ../MatlabHFM.cpp' ...
    ' COMPFLAGS="-std:c++17"' ... % needed on windows 
    ' -outdir ' binary_Dir ...
    ' ' cxxFlags  '-D' flag ...
    ' -I' '../../../Headers' ...
    ' -I' '../../../JMM_CPPLibs' ...
    ]);

compileModelsHFM = @(binary_Dir,modelNames) ...
    cellfun(@(name) compileHFM(binary_Dir,name,['ModelName=' name]), modelNames);

% 'Diagonal2','Diagonal3', temporally incompatible
standardModelsHFM = {'Isotropic2','Isotropic3',...
    'Riemann2','Riemann3','ReedsShepp2','ReedsSheppForward2','Elastica2',...
    'Dubins2','ReedsShepp3','ReedsSheppForward3'};

% Experimental models involved in some of the examples
% 'RiemannDiff2', temporally incompatible
%experimentalModelsHFM = {'IsotropicDiff2','RiemannLifted2_Periodic','DubinsExt2'};
experimentalModelsHFM = {'Seismic2','SeismicTopographic2','Seismic3','SeismicTopographic3','AlignedBillard','Riemann4',...
'Riemann5','Elastica2_9','ElasticaExt2_5','ReedsSheppExt2','ReedsSheppForwardExt2',...
'RiemannLifted2_Periodic','Rander2',...
'AsymmetricQuadratic2','AsymmetricQuadratic3','AsymmetricQuadratic3p1','Riemann3_Periodic'};

fprintf(['\nPlease execute the function compileModelsHFM(binary_Dir,standardModelsHFM) to build \n'...
'the Hamilton Fast Marching executables in directory binary_Dir, \n' ...
'In case of need, replace standardModelsHFM with experimentalModelsHFM, or any list of desired models.\n']);

%For me : binary_Dir = '/Users/mirebeau/Dropbox/Programmes/MATLAB/MexBin';
%For debug : compileHFM(binary_Dir,'Custom','Custom')
