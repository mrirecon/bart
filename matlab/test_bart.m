function test_bart()
%TEST_BART  Runs a unit test for the MATLAB BART wrapper.
%   TEST_BART() can be used to test if the BART toolbox is properly set-up
%   and if changes/additions made to the MATLAB wrapper break any core
%   functionality of the MATLAB wrapper.
%   
% Copyright 2020. Martin Krämer (Martin.Kraemer@uni-jena.de)
% 2022 Jon Tamir <jtamir.utexas.edu>

    %% Test setup
    testLog = [];
    capture = false;
    tolFloat = 1e-7;

    %% Test1: Environmental variable
    bartPath = get_bart_path()
    testAssert(~isempty(bartPath), 'BART path');

    %% Test2: Write/Read cfl
    file = tempname;
    data = rand(32,24,16);
    testRun('writecfl(file, data)','Write cfl file');
    dataRead = testRun('readcfl(file)','Read cfl file', 1);
    testAssert(~any(reshape(abs(data-dataRead{1}),[],1) > tolFloat), 'Data consistency cfl file');
    if (exist(strcat(file,'.cfl'),'file'))
        delete(strcat(file,'.cfl'))
    end
    
    %% Test3: Run bart with various parameters
    testRun('bart', 'Wrapper (without parameter)');
    testRun('bart traj -h', 'Wrapper (method help)');
    
    phantom = testRun("bart('phantom')", "Wrapper (No input, no parameter)", 1);
    testAssert(~isempty(phantom{1}), "Wrapper (No input, no parameter) - check output");
    
    phantom = testRun("bart('phantom -3')", "Wrapper (No input)", 1);
    testAssert(~isempty(phantom{1}), "Wrapper (No input) - check output)");
    
    phantom_kSpace = testRun("bart('fft -u 3', varargin{1})", "Wrapper (One input, one parameter)", 1, phantom{1});
    testAssert(~isempty(phantom_kSpace{1}), "Wrapper (One input, one parameter) - check output)");
    
    %% Check final test score
    failCount = sum(cellfun(@(x)(~x),testLog(:,2)));
    if (failCount == 0)
        fprintf('\nTEST Result: All Tests Passed!\n\n'); 
    else
        fprintf(2, '\nTEST Result: %i Tests Failed!\n\n', failCount); 
    end
    
    %% Helper functions
    function [Result] = testRun(Command, Name, OutCount, varargin)
        if (nargin < 3)
            OutCount = [];
            Result = [];
        end
        fprintf('TEST [%s] - running "%s" ', Name, Command);        
        
        status = false;
        try
            % when not printing to console (capture = true) we use evalc,
            % otherwise eval is used
            if (capture)
                fprintf('\n');
                if (isempty(OutCount))
                    eval(Command);
                else % to actually capture and return the output we have 
                     % pre initialize the results cell array with the
                     % pre-defined number of outputs to capture
                    Result = cell(OutCount);
                    [Result{:}] = eval(Command);
                end
            else
                if (isempty(OutCount))
                    evalc(Command);
                else
                    Result = cell(OutCount);
                    [~, Result{:}] = evalc(Command);
                end
            end
            status = true;
        catch
        end
        
        testLog = cat(1, testLog, {Name, status});
        fprintf(2 - status, '>> %s\n', testStatusToString(status));
    end
    
    function testAssert(Condition, Name)
        fprintf('TEST [%s] ', Name);
        
        testLog = cat(1, testLog, {Name, Condition});
        fprintf(2 - Condition, '>> %s\n', testStatusToString(Condition));
    end

    function [StatusString] = testStatusToString(Status)
        if (Status)
            StatusString = 'Passed';
        else
            StatusString = 'Failed';
        end
    end

    function printLog(log)
        for iLog = 1:size(log,1)
            fprintf('%s: %s\n', log{iLog, 1}, log{iLog, 2});
        end
    end
end
