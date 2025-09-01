within ;
package Credibility "Credibility Library - Version 0.2.0"
  extends Modelica.Icons.Package;
  import Modelica.Units;

  annotation (
    preferredView="info",
    version="0.2.0",
    versionDate="2024-06-13",
    dateModified = "2024-05-03 14:00:00Z",
    revisionId="$Format:%h %ci$",
    uses(
      Modelica(version="4.0.0")),
    Documentation(
      info="<html>
<p>
Package <strong>Credibility</strong> is a&nbsp;free Modelica package
to add traceability, uncertainty and calibration information to parameters in 
a standardized way. For details, see 
</p>

<ul>
<li><a href=\"modelica://Credibility.Examples\">Examples</a>, </li>
<li><a href=\"modelica://Credibility.UsersGuide\">User&apos;s Guide</a>, and the article </li>
<li> M. Otter, M. Reiner, J. Tobol&aacute;&rcaron;, L. Gall and M. Sch&auml;fer,
&quot;Towards Modelica Models with Credibility Information&quot;,
<em>MDPI Electronics</em>. 2022; 11(17):2728,.
<a href=\"https://doi.org/10.3390/electronics11172728\">doi 10.3390/electronics11172728</a>
</li>
</ul>

<p>
For copyright and BSD 3-clause license, see
<a href=\"modelica://Credibility.UsersGuide.License\">Copyright and License agreement</a>.
</p>
</html>",
      revisions="<html>
<table border=\"0\" cellspacing=\"0\" cellpadding=\"0\">
  <tr><td valign=\"middle\"> <img width=\"72\" src=\"modelica://Credibility/Resources/Images/DLR_Signet_schwarz.png\" alt=\"Logo DLR\"></td>
      <td valign=\"middle\"> <strong>Copyright &copy; DLR Institut für Systemdynamik und Regelungstechnik</strong> </td>
  </tr>
 </table>
<p>&nbsp;</p>
</html>"),
    Icon(graphics={
        Polygon(
          points={{-56,-2},{-20,-56},{62,38},{48,52},{-20,-20},{-40,10},{-56,-2}},
          lineColor={0,140,72},
          fillColor={0,140,72},
          fillPattern=FillPattern.Solid)}));
end Credibility;
