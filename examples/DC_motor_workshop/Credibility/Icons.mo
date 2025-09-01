within Credibility;
package Icons "Icons used in the library"
  extends Modelica.Icons.IconsPackage;


  partial record CredibleBase "Basic icon for Credibility records"

    annotation (
      Documentation(info="<html>
<p>
This icon indicates a record being used to define credibility attributes.
</p>
</html>"),
      Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{100,100}},
          grid={1,1}), graphics={
          Text(
            textColor={0,0,255},
            extent={{-150,50},{150,90}},
            textString="%name"),
          Rectangle(
            extent={{-100,-60},{100,-10}},
            lineColor={0,140,72},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
          Rectangle(
            extent={{-100,40},{100,-10}},
            lineColor={0,140,72},
            fillColor={0,140,72},
            fillPattern=FillPattern.Solid),
          Text(
            extent={{-50,40},{100,-10}},
            textColor={255,255,255},
            textString="credible"),
          Polygon(
            points={{-96,14},{-84,-8},{-56,32},{-64,37},{-83,11},{-90,20},{-96,14}},
            lineColor={0,140,72},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid)}),
      Diagram(coordinateSystem(grid={1,1})));
  end CredibleBase;
  annotation (Documentation(info="<html>
<p>
This package contains definitions for the graphical layout of components which may
be used in different packages. The icons can be utilized by inheriting them in the
desired class using &quot;extends&quot; or by directly copying the &quot;icon&quot; layer.
</p>
</html>"));
end Icons;
