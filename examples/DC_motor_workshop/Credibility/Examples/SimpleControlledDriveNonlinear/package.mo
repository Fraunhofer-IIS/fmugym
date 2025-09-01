within Credibility.Examples;
package SimpleControlledDriveNonlinear "Collection of examples involving simple drive train"
  extends Modelica.Icons.ExamplesPackage;

  annotation (
    Documentation(info="<html>
<p>
This package contains an example of a&nbsp;simple 1-dim. drive train and
all classes demonstrating a&nbsp;possible usage of parameter credibility
information within the context of this drive train. Most important:
</p>

<p>
<a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.PartialDrive\">PartialDrive</a>
is a&nbsp;partial model of a&nbsp;drive train with credibility information. 
The structure can be seen in the following image (from
[<a href=\"modelica://Credibility.UsersGuide.References\">Otter2022</a>]):
</p>

<div>
<img src=\"modelica://Credibility/Resources/Images/CredibilityLibrary1.png\" alt=\"A screenshot
of Modelica library Credibility together with parametrization of a drive train example\">
</div>

<p>
<a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.SimpleControlledDrive_original\">SimpleControlledDrive_original</a>
uses the partial model of the drive train, adds a&nbsp;controller and concrete
values including credibility information by replacing the <em>PartialData</em> record with 
<a href=\"modelica://Credibility.Examples.SimpleControlledDriveNonlinear.DataVariant001\">DataVariant001</a>.
In particular nominal, upper/lower limit values for the table of the gear
compliance are provided
(image from [<a href=\"modelica://Credibility.UsersGuide.References\">Otter2022</a>]):
</p>

<div>
<img src=\"modelica://Credibility/Resources/Images/CredibilityLibrary2.png\" alt=\"A screenshot
of component Credibility.Examples.SimpleControlledDriveNonlinear.SimpleControlledDrive_original\">
</div>
</html>"));
end SimpleControlledDriveNonlinear;
