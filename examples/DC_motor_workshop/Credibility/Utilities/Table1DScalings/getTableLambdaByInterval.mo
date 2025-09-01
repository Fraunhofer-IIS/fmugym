within Credibility.Utilities.Table1DScalings;
function getTableLambdaByInterval "Calculate scaled table by lower and upper limits"
  extends Modelica.Icons.Function;

  input Real lambda "Convex scaling between -1 and 1 (=0: nominal)";
  input Real uncertainty[:,4] "Table uncertainty (x, y, yL, yU)";
  output Real table[size(uncertainty,1),2] "Value of table";
algorithm
  table[:,1] := uncertainty[:,1];

  if lambda >= 0 then
    table[:,2] := lambda * uncertainty[:,4] + (1 - lambda) * uncertainty[:,2];
  else
    table[:,2] := (1 + lambda) * uncertainty[:,2] - lambda * uncertainty[:,3];
  end if;

  annotation (Documentation(info="<html>
<p>
Evaluate <code>table</code> from the <code>uncertainty</code> matrix
by a&nbsp;convex combination utilizing attribute <code>lambda</code>.
Hereby, the matrix <code>uncertainty</code> contains
</p>
<ul>
  <li>
    col. 1: grid values,
  </li>
  <li>
    col. 2: vector of <em>nominal values</em>,
  </li>
  <li>
    col. 3: vector of <em>lower</em> limits and
  </li>
  <li>
    col. 4: vector of <em>upper</em> limits.
  </li>
</ul>
<p>
The convex combination is defined for the Real <code>lambda</code> in a&nbsp;way that 
</p>
<ul>
  <li>
    <var>&lambda;</var> = &minus;1 results in
    <code>table[:,&nbsp;2]</code>&nbsp;= <code>uncertainty[:,&nbsp;3]</code>,
  </li>
  <li>
    <var>&lambda;</var> = 0 results in
    <code>table[:,&nbsp;2]</code>&nbsp;= <code>uncertainty[:,&nbsp;2]</code> and
  </li>
  <li>
    <var>&lambda;</var> = 1 results in
    <code>table[:,&nbsp;2]</code>&nbsp;= <code>uncertainty[:,&nbsp;4]</code>.
  </li>
</ul>

<p>
This function is called in <a href=\"modelica://Credibility.Table1D\">Table1D</a>
to evaluate &quot;table[i,j]&quot;.
For further details, refer to 
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.UncertaintyInfo\">User&apos;s Guide</a>
&ndash; &quot;Convex scaling by uncertain parameter <var>&lambda;</var>&quot;.
</p>
</html>"));
end getTableLambdaByInterval;
