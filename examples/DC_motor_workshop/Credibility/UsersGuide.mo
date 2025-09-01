within Credibility;
package UsersGuide "User's Guide"
  extends Modelica.Icons.Information;

  package ParameterCredibility "Parameter credibility"
    extends Modelica.Icons.Information;

    class TraceabilityInfo "Traceability information"
      extends Modelica.Icons.Information;

      annotation (Documentation(info="<html>
<p>
The aim of <em>traceability</em> is that anyone working with the model knows
at any time why a&nbsp;parameter has a&nbsp;specific value. This is defined
in the library with the three keys:
</p>
<ul>
  <li>
    <strong>source</strong>: origin of the parameter value (e.g. <em>Estimated</em>),
    see below.
  </li>
  <li>
    <strong>info</strong> (optional): a&nbsp;short textual explanation how the value
    was determined.
  </li>
  <li>
    <strong>reference</strong> (optional): identifies the resource where the
    determination of the value is described, for example web-address, DOI, ISBN,
    internal report ID of a&nbsp;document, data-sheet etc.
    The referenced resource must be stored in a&nbsp;way that avoids accidental
    manipulation, e.g., in a&nbsp;version control system with a&nbsp;given hash,
    in order to retain the reference information reliably. 
  </li>
</ul>

<h4>Source</h4>
<p>
It is the way, how the value of the parameter was determined, and is realized as
an enumeration
<a href=\"modelica://Credibility.Types.SourceType\">SourceType</a>
that can take one of the following values.
</p>
<ul>
  <li>
    <strong>Unknown</strong>: the source of the value is not known. Such parameters
    have to be handled with care and are not sufficient for the demand of a&nbsp;good
    traceability
  </li>
  <li>
    <strong>Estimated</strong>: value from an educated guess, previous projects,
    extrapolated value, etc. 
  </li>
  <li>
    <strong>Provided</strong>: value from a&nbsp;data sheet, supplier specification, etc.
    The <em>reference</em> (see above) to this source shall be also provided.
  </li>
  <li>
    <strong>Computed</strong>: value from computer aided data such as CAD, CFD, FEM, etc.
    The <em>reference</em> (see above) to this source shall be also provided.
  </li>
  <li>
    <strong>Measured</strong>: measured value from physical measurements.
    The <em>reference</em> (see above) to this source shall be also provided.
  </li>
  <li>
    <strong>Calibrated</strong>: value from a&nbsp;model calibration (based on provided
    data). The setup of the calibration &ndash; this means all information needed to
    repeat exactly this calibration &ndash; and the allowed parameter range
    should be stored in the calibration section of the parameter record, see
    <a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.CalibrationInfo\">calibration information</a>.
  </li>
</ul>
</html>"));
    end TraceabilityInfo;

    class UncertaintyInfo "Uncertainty information"
      extends Modelica.Icons.Information;

      annotation (Documentation(info="<html>
<p>
This is the information about the quality of a&nbsp;parameter value and, thus,
an essential part of a&nbsp;credible model. Besides a&nbsp;mathematical
description of the uncertainty (see e.g.
[<a href=\"modelica://Credibility.UsersGuide.References\">Otter2022</a>]), also
the same information as for the
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.TraceabilityInfo\">traceability
of parameter value</a> needs to be given:
</p>
<ul>
  <li>
    <strong>source</strong>: origin of the uncertainty determination (e.g.
    <em>Estimated</em>), see
    <a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.TraceabilityInfo\">traceability</a>.
  </li>
  <li>
    <strong>info</strong> (optional): a&nbsp;short textual explanation how the
    uncertainty was determined.
  </li>
  <li>
    <strong>reference</strong> (optional): identifies the resource where the
    uncertainty determination is described, for example web-address, DOI, ISBN,
    internal report ID of a&nbsp;document, data-sheet etc.
    The referenced resource must be stored in a&nbsp;way that avoids accidental
    manipulation, e.g., in a&nbsp;version control system with a&nbsp;given hash,
    in order to retain the reference information reliably. 
  </li>
</ul>

<p>
Additionally, the mathematical description of uncertainties of scalars and of
arrays/tables must be provided. Particular descriptions provided with
the library are collected in package 
<a href=\"modelica://Credibility.Types.Uncertainty\">Types.Uncertainty</a>.
</p>
<p>
The treatment implemented in the library is focused on providing the information
in a&nbsp;way that it can be used by models described with the Modelica language,
<a href=\"https://fmi-standard.org\">FMI</a> or <a href=\"https://ssp-standard.org\">SSP</a>.
Furthermore, not only physical/measured quantities are taken into account, but
the same description form is also used to define requirements or specifications.
For example, a&nbsp;simulation result must match a&nbsp;reference result with some
uncertainty, or the result of a&nbsp;design must match some criteria that is
specified with an uncertainty description.
</p>

<p>
Every physical quantity has inherent limits. Therefore, the upper and lower limits
of a&nbsp;scalar value need to be defined, independent of the kind of the mathematical
description of the uncertainty. For this reason, the minimum information to be
provided for a&nbsp;scalar parameter with an uncertainty description of any kind is:
</p>

<ul>
  <li>
    nominal &ndash; the nominal value of the scalar (e.g., determined by calibration).
  </li>
  <li>
    lower &ndash; the lowest possible value of the uncertain scalar.
  </li>
  <li>
    upper &ndash; the highest possible value of the uncertain scalar.
  </li>
</ul>

<p>
Often, it is inconvenient to provide absolute ranges and instead, relative or absolute
deviations are more practical. In the
<a href=\"https://emphysis.github.io/pages/downloads/efmi_specification_1.0.0-alpha.4.zip\">eFMI standard</a> 
(Section 3.2.4) tolerances for reference results are defined in a&nbsp;similar
way as tolerances for numerical integration algorithms. Due to its generality,
this description form of eFMI is used here as well:
</p>

<ul>
  <li>
    nominal &ndash; nominal value of the scalar
  </li>
  <li>
    relTol &ndash; relative tolerance of limits with respect to nominal (default = 0.0).
  </li>
  <li>
    absTol &ndash; absolute tolerance of limits with respect to nominal (default = 0.0).
  </li>
</ul>

<p>
The lower and upper values are computed from these parameters in the following way:
</p>

<blockquote><pre>
tol   = max(absTol, relTol*|nominal|)
lower = nominal &minus; tol
upper = nominal + tol
</pre></blockquote>

<h5>Example</h5>
<p>
The resistance of an electrical resistor can be defined using following tolerances:
</p>

<ul>
  <li>
    200&nbsp;&ohm;&nbsp;&plusmn;&nbsp;5&nbsp;% &ndash; limits defined with
    relTol&nbsp;=&nbsp;0.05 (absTol&nbsp;= 0&nbsp;&ohm;).
  </li>
  <li>
    200&nbsp;&ohm;&nbsp;&plusmn;&nbsp;10&nbsp;&ohm; &ndash; limits defined with
    absTol&nbsp;= 10&nbsp;&ohm; (relTol&nbsp;=&nbsp;0).
  </li>
</ul>


<p>
There is a&nbsp;huge literature on the mathematical description of uncertainties.
[<a href=\"modelica://Credibility.UsersGuide.References\">Riedmaier2021</a>]
provides a&nbsp;comprehensive literature overview.
In [<a href=\"modelica://Credibility.UsersGuide.References\">Bouskela2011</a>],
some constructs are proposed to describe uncertain values in the Modelica language. 
In the Credibility Library, the following uncertainty descriptions are currently
provided for scalar Real parameters, see
<a href=\"modelica://Credibility.Types\">Types</a>:
</p>

<table border=\"1\" cellspacing=\"0\" cellpadding=\"2\">
  <caption align=\"bottom\">
    Examples to define uncertainty descriptions with generic and tolerance
    parameterizations of scalar parameters.
    Default values are given with &quot;=...&quot;, such as &quot;relTol=0&quot;.
    A&nbsp;truncated normal distribution is defined by the <code>nominal</code>
    value (= mean value) and the standard deviation <code>stdDev</code> of the
    underlying normal distribution, besides <code>lower</code> and
    <code>upper</code> limits.
  </caption>
  <tr>
    <th>
      Uncertainty name
    </th>
    <th>
      Parameters
    </th>
    <th>
      Uncertainty type
    </th>
  </tr>
  <tr>
    <td>
      <a href=\"modelica://Credibility.Types.Interval\">Interval</a>
    </td>
    <td>
      <code>nominal</code>, <code>lower</code>, <code>upper</code>
    </td>
    <td>
      epistemic
    </td>
  </tr>
  <tr>
    <td>
      <a href=\"modelica://Credibility.Types.Uniform\">Uniform</a>
    </td>
    <td>
      <code>nominal</code>, <code>lower</code>, <code>upper</code>
    </td>
    <td>
      uniform distribution
    </td>
  </tr>
  <tr>
    <td>
      <a href=\"modelica://Credibility.Types.TruncatedNormal\">TruncatedNormal</a>
    </td>
    <td>
      <code>nominal</code>, <code>lower</code>, <code>upper</code>, <code>stdDev</code>
    </td>
    <td>
      truncated normal distribution
    </td>
  </tr>
  <tr>
    <td>
      <a href=\"modelica://Credibility.Types.IntervalTolerance\">IntervalTolerance</a>
    </td>
    <td>
      <code>nominal</code>, <code>relTol=0</code>, <code>absTol=0</code>
    </td>
    <td>
      epistemic
    </td>
  </tr>
  <tr>
    <td>
      <a href=\"modelica://Credibility.Types.UniformTolerance\">UniformTolerance</a>
    </td>
    <td>
      <code>nominal</code>, <code>relTol=0</code>, <code>absTol=0</code>
    </td>
    <td>
      uniform distribution
    </td>
  </tr>
  <tr>
    <td>
      <a href=\"modelica://Credibility.Types.TruncatedNormalTolerance\">TruncatedNormalTolerance</a>
    </td>
    <td>
      <code>nominal</code>, <code>relTol=0</code>, <code>absTol=0</code>,
      <code>stdDevFactor=3</code>
    </td>
    <td>
      truncated normal distribution
    </td>
  </tr>
</table>


<p>
An example for a&nbsp;TruncatedNormalTolerance uncertainty of the abovementioned
resistance is given in the next two figures: 
</p>

<div>
<img src=\"modelica://Credibility/Resources/Images/TruncatedNormalProbabilityDensity.svg\" alt=\"Probability
density of the truncated normal distribution of resistance R = 200 W plusminus 10 W, 2 sigma\">
<br>
<img src=\"modelica://Credibility/Resources/Images/TruncatedNormalCumulativeProbability.svg\" alt=\"Cumulative
probability of the truncated normal distribution of resistance R = 200 W plusminus 10 W, 2 sigma\">
</div>


<h4>Array uncertainty</h4>
<p>
Arrays occur in system modelling and analysis in various ways:
</p>
<ul>
  <li>
    Submodels are often approximated by tables of a&nbsp;characteristic property
    that have been determined by measurements, for example a&nbsp;table defining
    friction torque as function of relative angular velocity, or a&nbsp;table
    defining mass flow rate through a&nbsp;valve as function of valve position.
    Tables of this kind are basically defined with two or more dimensional arrays.
  </li>
  <li>
    Inputs of a&nbsp;system are often defined by tables as function of time.
  </li>
  <li>
    Outputs of a&nbsp;system might be checked against reference outputs (for
    example, determined from the previous version of the model or tool, from
    another tool or from an analytically known solution) defined by tables as
    function of time.
    The computed solution is then required to be within uncertainty ranges of
    the reference tables.
  </li>
</ul>
<p>
All these examples have parameter arrays, where the uncertainties
of the array elements need to be described.
</p>

<p>
If a&nbsp;characteristic is determined by detailed measurements, then typically
for every element upper and lower limits are known from the available measurement
data.
This means that the measurements can be summarized by a&nbsp;<em>nominal array</em>
that has been determined by the calibration process and arrays lower and upper
of the same size that define <em>lower</em> and <em>upper</em> limits, so all
measured data of the respective characteristics is between these limits.
Furthermore, it must be defined how to interpolate between the table values,
given the vectors of the independent variables (an <var>n</var>-dimensional table
is defined by <var>n</var>&nbsp;vectors of independent variables).
Often, linear interpolation is used for tables derived from measurements.
</p>
<p>
An envisaged uncertainty description with <em>generic</em> parameterization of
2-dim. array can be demonstrated in the following example.
The vectors of the two independent variables are <var>u1</var>&nbsp;= [0.1, 0.2, 0.3, 0.4]
and <var>u2</var>&nbsp;= [10, 20, 30]. The nominal array is a&nbsp;matrix of size
[4,&nbsp;3] and, e.g., of values [110 120 130; 210 220 230; 310 320 330; 410 420 430]
(i.e. <var>y</var>&nbsp;= 230 is the nominal value for <var>u1</var>&nbsp;= 0.2 and
<var>u2</var>&nbsp;= 30). Both the lower and upper limits are matrices of size
[4,&nbsp;3] as well and have exemplary values
<var>lower</var>&nbsp;= [11 12 13; 21 22 23; 31 32 33; 41 42 43] and
<var>upper</var>&nbsp;= [119 129 139; 219 229 239; 319 329 339; 419 429 439].
Thus, e.g. the uncertain table value of <var>u1</var>&nbsp;= 0.2 and <var>u2</var>&nbsp;= 30
(i.e. array element [2,&nbsp;3]) must be in the range [23&nbsp;&hellip;&nbsp;239].
</p>


<h5>Convex scaling by uncertain parameter <var>&lambda;</var></h5>
<p>
The question arises how a&nbsp;parameter array can be used to compute the uncertainty
distributions of output variables. For <em>scalar parameters</em>, the standard Monte
Carlo simulation method is based on the approach to draw sufficient numbers of random
numbers according to the respective distributions and for each set of selected parameter
values perform a&nbsp;simulation.
Such a&nbsp;procedure might not be directly applicable to <em>parameter arrays</em>,
because randomly selected elements of an array might give a&nbsp;table that does not
represent physics (for example, if the table output is a&nbsp;monotonically increasing
function of the input, then this property might be lost if table elements are randomly
selected).
</p>
<p>
For simplicity, the following method is used for 1-dim. tables in the Library.
A&nbsp;generalization to multi-dimensional tables is straightforward. A&nbsp;1-dim.
table defines a&nbsp;function <var>y</var>&nbsp;= <var>f</var>&nbsp;(<var>u</var>&nbsp;)
that computes the output&nbsp;<var>y</var> from the input&nbsp;<var>u</var> by
interpolation in a&nbsp;table, cf. 
<a href=\"modelica://Modelica.Blocks.Tables.CombiTable1Ds\">Modelica.Blocks.Tables.CombiTable1Ds</a>.
The table is defined as &quot;table[i,j]&quot; 
with the first column &quot;table[:,1]&nbsp;= u[:]&quot; which contains the grid
points&nbsp;<var>u<sub>i</sub></var>, and the second column &quot;table[:,2]&nbsp;= y[:]&quot;
which contains the data&nbsp;<var>y<sub>i</sub></var> to be interpolated.
This &quot;table[i,j]&quot; is computed from the given matrix &quot;uncertainty&quot;
by a&nbsp;convex combination utilizing uncertain parameter <var>&lambda;</var>.
Hereby, the matrix &quot;uncertainty&quot; contains
</p>
<ul>
  <li>
    grid values &quot;uncertainty[:,1]&nbsp;= u[:]&quot;,
  </li>
  <li>
    vector of <em>nominal values</em> &quot;uncertainty[:,2]&nbsp;= nominal[:]&quot;,
  </li>
  <li>
    vector of <em>lower</em> limits &quot;uncertainty[:,3]&nbsp;= lower[:]&quot; and
  </li>
  <li>
    vector of <em>upper</em> limits &quot;uncertainty[:,4]&nbsp;= upper[:]&quot;.
  </li>
</ul>
<p>
The convex combination is defined in a&nbsp;way that <var>&lambda;</var>&nbsp;=&nbsp;&minus;1 results
in &quot;y[i]&nbsp;= lower[i]&quot;, <var>&lambda;</var>&nbsp;=&nbsp;0 results in
&quot;y[i]&nbsp;= nominal[i]&quot; and <var>&lambda;</var>&nbsp;=&nbsp;1 results in
&quot;y[i]&nbsp;= upper[i]&quot;, in particular:
</p>
<div>
<img src=\"modelica://Credibility/Resources/Images/equationConvexCombination.png\" alt=\"Equation to calculate y[i] by convex combination utilizing lambda\"/>
</div>

<p>
A&nbsp;realization can be seen in <a href=\"modelica://Credibility.Table1D\">Credibility.Table1D</a>
where &quot;uncertainty&quot; table is in fact defined by parameter <code>uncertainty.table</code>.
In this record, the abovementioned calculation is done in the function 
<a href=\"modelica://Credibility.Utilities.Table1DScalings.getTableLambdaByInterval\">getTableLambdaByInterval</a>.
There is also implemented an alternative calculation of &quot;table[i,j]&quot; called
<a href=\"modelica://Credibility.Utilities.Table1DScalings.getTableLambdaByTolerance\">getTableLambdaByTolerance</a>
utilizing &ndash; either relative or absolute &ndash;
tolerances.
</p>
</html>"));
    end UncertaintyInfo;

    class CalibrationInfo "Calibration information"
      extends Modelica.Icons.Information;

      annotation (Documentation(info="<html>
<p>
Once a&nbsp;model is implemented, the modeling process goes on with the
calibration, validation and verification, to ensure that the model appropriately
achieves the aims it was made for. Since the scope of model calibration,
validation and verification is very large, we can only give a&nbsp;short
overview in the context of a&nbsp;typical use case for a&nbsp;multi-physical
Modelica model.
<!--
For a&nbsp;broader overview on the topic, a&nbsp;recent paper [17] goes into more
details and gives many additional references.
-->
</p>
<p>
The aim of the calibration process is to parameterize models with the help of
measurements with regard to defined goals and criteria. For models, this generally
means that they should map the behavior of the real-world system as precisely as
possible. The calibration of the parameters of these models is important in order
to achieve a&nbsp;good representation of the real system or to have a&nbsp;good
controller performance, in the case of model-based control.
</p>
<p>
As discussed in 
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.TraceabilityInfo\">traceability section</a>,
for many physical systems the knowledge of the individual involved parameters can
be quite different. For example, for the model of a&nbsp;robot, the mechanical
parameters such as link lengths or masses could be known very precisely from
data-sheets or CAD data, whereas the friction or damping in connecting joints can
be highly unknown. Well known parameters should, thus, be included in the models
directly, in order to reduce the number of unknown parameters for the calibration
process. Additionally, the source of these parameters should be well documented, cf.
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.TraceabilityInfo\">traceability</a>.
</p>
</html>"));
    end CalibrationInfo;

    annotation (Documentation(info="<html>
<p>
For a&nbsp;credible model it is not sufficient to store only information about
the whole model, e.g. using metadata, but also a&nbsp;documentation of each
parameter, including credibility information, should be available.
In this library, it is introduced such type of credibility information for
model parameters. According to implemented approach, each parameter of
a&nbsp;Modelica model does not only define a&nbsp;scalar or an array value,
but the <em>parameter value</em> and additional information concerning
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.TraceabilityInfo\">traceability</a>,
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.UncertaintyInfo\">uncertainty</a>
and
<a href=\"modelica://Credibility.UsersGuide.ParameterCredibility.CalibrationInfo\">calibration</a>
are stored in a&nbsp;record data structure. An example for a compliance spring constant
is given in the next figure (from
[<a href=\"modelica://Credibility.UsersGuide.References\">Otter2022</a>]):
</p>

<div>
<img src=\"modelica://Credibility/Resources/Images/SpringConstantCredibility.png\" alt=\"Credibility attributes of spring constant c\">
</div>
</html>"));
  end ParameterCredibility;

  class ReleaseNotes "Release notes"
    extends Modelica.Icons.ReleaseNotes;

    annotation (
      Documentation(
        info="<html> 
<p>
This section summarizes the changes that have been performed on the library.
</p>

<h4>Version 0.2.0 (2024-06-13)</h4>
<p>
The following improvements have been realized.
</p>
<ul>
  <li>
    Functions defined in the protected section of record
    <a href=\"modelica://Credibility.Table1D\">Table1D</a>
    moved into package
    <a href=\"modelica://Credibility.Utilities.Table1DScalings\">Utilities.Table1DScalings</a>.
  </li>
  <li>
    <a href=\"modelica://Credibility.Table1D\">Table1D</a>: fix mismatching array size of
    <code>uncertainty.table</code> in the function call
    <a href=\"modelica://Credibility.Utilities.Table1DScalings.getTableLambdaByTolerance\">getTableLambdaByTolerance</a>().
  </li>
  <li>
    Example
    <a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.SimpleControlledDrive_virtualMeasurement\">SimpleControlledDrive_virtualMeasurement</a>:
    Use &quot;fixedLocalSeed&quot; for noise generators to always get reproducible results.
  </li>
  <li>
    Enable multilingual descriptions and add German translation. See also Modelica Specification &ndash;
    <a href=\"https://specification.modelica.org/maint/3.6/packages.html#multilingual-descriptions\">Multilingual Descriptions</a>.
  </li>
</ul>

<h4>Version 0.1.0 (2022-09-17)</h4>
<ul>
  <li>
    First official release of the library.
  </li>
</ul>
</html>",
        revisions="<html>
</html>"));
  end ReleaseNotes;

  class References "References"
    extends Modelica.Icons.References;

    annotation (
      Documentation(
        info="<html>
<p>
The following books and articles have been useful for 
the implementation of the library.
</p>
 
<table border=\"0\" cellspacing=\"0\" cellpadding=\"2\">
<!--
  <tr>
    <td>
      [KeyYear]
    </td>
    <td>
      Author1 and Author2,
      &quot;Article title,&quot;
      <em>Journal</em>,
      information, year.
    </td>
  </tr>
-->
  <tr>
    <td>
      [Otter2022]
    </td>
    <td>
      M. Otter, M. Reiner, J. Tobol&aacute;&rcaron;, L. Gall and M. Sch&auml;fer,
      &quot;Towards Modelica Models with Credibility Information&quot;,
      <em>Electronics</em>. 2022; 11(17):2728.
      <a href=\"https://doi.org/10.3390/electronics11172728\">doi 10.3390/electronics11172728</a>
    </td>
  </tr>
  <tr>
    <td>
      [Gall2021]
    </td>
    <td>
      L. Gall, M. Otter, M. Reiner, M. Sch&auml;fer and J. Tobol&aacute;&rcaron;,
      &quot;Continuous Development and Management of Credible Modelica Models&quot;,
      <em>14th International Modelica Conference</em>,
      pp. 359&ndash;372, <a href=\"https://doi.org/10.3384/ecp21181359\">doi 10.3384/ecp21181359</a>,
      2021.
    </td>
  </tr>
  <tr>
    <td>
      [Riedmaier2021]
    </td>
    <td>
      S. Riedmaier, B. Danquah, B. Schick and F. Diermeyer,
      &quot;Unified Framework and Survey for Model Verification, Validation and Uncertainty Quantification&quot;,
      <em>Archives of Computational Methods in Engineering</em>,
      pp- 2655&ndash;2688, Springer, <a href=\"https://doi.org/10.1007/s11831-020-09473-7\">doi 10.1007/s11831-020-09473-7</a>,
      2021.
    </td>
  </tr>
  <tr>
    <td>
      [Bouskela2011]
    </td>
    <td>
      D. Bouskela, A. Jardin, Z. Benjelloun-Touimi, P. Aronsson and P. Fritzson,
      &quot;Modelling of Uncertainties with Modelica,&quot;
      <em>8th Modelica Conference</em>,
      pp. 673&ndash;685, <a href=\"https://doi.org/10.3384/ecp11063673\">doi 10.3384/ecp11063673</a>,
      2011.
    </td>
  </tr>
</table> 
</html>",
        revisions="<html>
</html>"));
  end References;

  class Contact "Contact"
    extends Modelica.Icons.Contact;

    annotation (
      Documentation(
        info="<html>
<h4>Development</h4>
<blockquote>
  Martin Otter (head of development)<br>
  Deutsches Zentrum f&uuml;r Luft- und Raumfahrt e.V. (DLR)<br>
  Institut f&uuml;r Systemdynamik und Regelungstechnik (SR)<br> 
  M&uuml;nchener Stra&szlig;e 20<br>
  D-82234 We&szlig;ling<br>
  E-mail: <a href=\"mailto:sr-modelica@dlr.de\">sr-modelica@dlr.de</a><br>
  Web:    <a href=\"https://rmc.dlr.de/sr/de/staff/martin.otter/\">rmc.dlr.de/sr/de/staff/Martin.Otter</a>
</blockquote>
 
 
<h4>Contributors</h4>
<ul>
  <li>Matthias Reiner (DLR-SR)</li>
  <li>Jakub Tobolar (DLR-SR)</li>
</ul>
 
 
<h4>Acknowledgement</h4>
<p>
The development of the Library was organized within the European ITEA3
Call6 project <a href=\"https://www.upsim-project.eu\">UPSIM</a> &ndash;
Unleash Potentials in Simulation (number 19006). The work was partially
funded by the German Federal Ministry of Education and Research (BMBF,
grant numbers 01IS20072H and 01IS20072G).
</p>
<p>
The development of this library is based on the work that was carried out
by the library authors (see above) and by Leo Gall and Matthias Sch&auml;fer (both
<a href=\"https://www.ltx.de\">LTX Simulation GmbH</a>) in the UPSIM project,
see journal article <em>Towards Modelica Models with Credibility Information</em>,
<a href=\"https://doi.org/10.3390/electronics11172728\">doi 10.3390/electronics11172728</a>.
</p>
<p>
We would like to thank our colleague Andreas Pfeiffer (DLR-SR) for his
idea about the convex combination approach used in  <a href=\"modelica://Credibility.Table1D\">Table1D</a>
as well as for his contributions during discussion and realization of the
model calibration of the simple drive line model.
</p>
</html>",
        revisions="<html>
</html>"));
  end Contact;

  class License "Copyright and License"
    extends Modelica.Icons.Information;

    annotation (
      Documentation(
        info="<html>
<p>
All files in this directory (<em>Credibility</em>) and in all
subdirectories, especially all files that build package \"Credibility\"
and all files in \"Credibility/Resources/\" and \"Credibility/help/\"
are licensed by <strong>DLR</strong> under the
<strong>3-Clause BSD License</strong>.
</p>

<h4>Licensor</h4>
<p>
Deutsches Zentrum f&uuml;r Luft- und Raumfahrt (DLR)<br>
<a href=\"https://www.dlr.de/sr/en\">Institut f&uuml;r Systemdynamik und Regelungstechnik (SR)</a><br>
M&uuml;nchener Stra&szlig;e 20<br>
D-82234 We&szlig;ling<br>
Germany
</p>

<h4>Copyright</h4>
<p>
&copy; 2022-2024, DLR Institut f&uuml;r Systemdynamik und Regelungstechnik
<br>All rights reserved.
</p>

<h4><a name=\"3clauseBSD_License-outline\"></a>The 3-clause BSD License</h4>
<p>
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
</p>
<p>
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
</p>
<p>
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
</p>
<p>
3. Neither the name of the copyright holder nor the names of its contributors may
be used to endorse or promote products derived from this software without specific
prior written permission.
</p>
<p>
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
</p>
---
<p>
The above is the license, and is the standard
<a href=\"https://opensource.org/licenses/BSD-3-Clause\">3-clause BSD-license</a>
with DLR Institut f&uuml;r Systemdynamik und Regelungstechnik as copyright holder.
</p>
</html>",
        revisions="<html>
</html>"));
  end License;
  annotation (
    DocumentationClass=true,
    Documentation(info="<html>
<p>
This is a&nbsp;short user&apos;s guide for the overall library.
</p>
</html>"));
end UsersGuide;
