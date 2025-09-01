within Credibility.Utilities.Table1DScalings;
function getTableLambdaByTolerance "Calculate scaled table by tolerances"
  extends Modelica.Icons.Function;

  input Real lambda "Convex scaling between -1 and 1 (=0: nominal)";
  input Real uncertainty[:,2] "Table uncertainty (x, nominal)";
  input Real absTol "Limits: |value[i] - uncertainty[i,2]| <= max(absTol,relTol*|uncertainty[i,2]|)";
  input Real relTol "Limits: |value[i] - uncertainty[i,2]| <= max(absTol,relTol*|uncertainty[i,2]|)";
  output Real table[size(uncertainty,1),2] "Value of table";
protected
  Real tol[size(uncertainty,1)];
algorithm
  table[:,1] := uncertainty[:,1];

  for i in 1:size(uncertainty,1) loop
    tol[i] :=max(absTol, relTol*abs(uncertainty[i, 2]));
  end for;
  if lambda >= 0 then
    table[:,2] := lambda * (uncertainty[:,2]+tol) + (1 - lambda) * uncertainty[:,2];
  else
    table[:,2] := (1 + lambda) * uncertainty[:,2] - lambda * (uncertainty[:,2]-tol);
  end if;

  annotation (Documentation(info="<html>
<p>
Evaluate <code>table</code> from the <code>uncertainty</code> matrix
and <strong>tolerances</strong> by a&nbsp;convex combination utilizing
attribute <code>lambda</code>.
Hereby, the matrix <code>uncertainty</code> contains
</p>
<ul>
  <li>
    col. 1: grid values,
  </li>
  <li>
    col. 2: vector of <em>nominal values</em>.
  </li>
</ul>
<p>
The convex combination is defined for the Real <code>lambda</code> in a&nbsp;way that 
</p>
<ul>
  <li>
    <var>&lambda;</var> = &minus;1 results in
    <code>table[:,&nbsp;2]</code>&nbsp;= <code>uncertainty[:,&nbsp;2] - tol[:]</code>,
  </li>
  <li>
    <var>&lambda;</var> = 0 results in
    <code>table[:,&nbsp;2]</code>&nbsp;= <code>uncertainty[:,&nbsp;2]</code> and
  </li>
  <li>
    <var>&lambda;</var> = 1 results in
    <code>table[:,&nbsp;2]</code>&nbsp;= <code>uncertainty[:,&nbsp;2] + tol[:]</code>.
  </li>
</ul>
<p>
The tolerance vector <code>tol</code> is defined element-wise by
</p>

<blockquote><pre>
tol[i] = max( absTol, relTol*abs(uncertainty[i, 2]) ).
</pre></blockquote>

<p>
This function is called in <a href=\"modelica://Credibility.Table1D\">Table1D</a>
to evaluate &quot;table[i,j]&quot;.
For further details, refer to 
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.UncertaintyInfo\">User&apos;s Guide</a>
&ndash; &quot;Convex scaling by uncertain parameter <var>&lambda;</var>&quot;.
</p>
</html>"));
end getTableLambdaByTolerance;
