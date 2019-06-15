for i=1:10
    lineOfCode = sprintf('A%d = [1:i]', i);
    % Now, execute lineOfCode just as if you'd typed 
    % >> Ai=[i:i]' into the command window
    eval(lineOfCode);
end