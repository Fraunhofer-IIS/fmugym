#ToDo: Variablennamen klären, so dass sie schöne Indizes im Paper ergeben (aktuell schon ganz ok)

import math

#Input-Toleranzfelder:
R1 = 0.1
S2 = 0.1
C4 = 0.1
S4 = 0.2
N2 = 0.1
C5 = 0.2
R4 = 0.1

#Steuergröße negativer Input
S4_negative = False

# --- S4 prüfen und flag für Rechnungen unten setzen ---

if S4 < 0:
    S4_negative = True
    S4 = -S4

#Länge rotatorische Tol-Felder
Char_L_S4 = 27.5    #Zylinderfläche Stator außen
Char_L_R4 = 17.2679 #Zylinderfläche Rotor Spindelfläche

#Geometrische Größen (alle in Figure 5 einzeichnen) + alle nochmal genau im CAD prüfen (Fehlerquelle)
# Allgemeines / Längen
l_R = 30.5    # Länge Rotor zwischen Kontaktpunkten zu B1/B2
r_B = 1       # Radius Kugel B2   
a_C = 10      # Länge Drehpunkt C zu Ebene mit konstanter x Vorgabe -> Position diese Ebene hier als Annahme einfach definiert, im Paper wird dann beschreiben wo der Punkt liegt, Ursprünglicher Wert unter der Annahme der konstante x-Wert wäre genau an Kontaktstelle S3-C2 (scheint CETOL aber nicht so zu machen) --> genauer Drehpunkt ist in CETOL unbekannt  
# Positionen für Vektordrehungen
# Pos FR1 relativ zu Drehpunkt O in R
Pos_FR1_R_x = 13.96
Pos_FR1_R_y = 7.25
# Pos FR2 relativ zu Drehpunkt O in R fest verbunden mit N
Pos_FR2_RN_x = 31.25
Pos_FR2_RN_y = 1.25
# Pos FR2 relativ zu Drehpunkt Mantelfläche Zentrum R4 (keine Relativbewegung von C in diesem Fall) in N
Pos_FR2_N_x = 13.21
Pos_FR2_N_y = 1.25
# Drehung bei Kippen S4: Beide Teile drehen sich, Abstand der Koordinaten FR2RN zu FR2C (Referenz) muss bestimmt werden, beteiligt: Kombi R+N und C
#Bezugspunkt für Messung:
# Pos P bzw. B relativ zu O als Ansatzpunkt für C
Pos_P_R_x = 31.5 # = l + r
Pos_P_R_y = 0
# Pos FR2 relativ zu Drehpunkt Mitte B2 beim Verdrehen C um B2/R
Pos_FR2ref_C_x = -0.25
Pos_FR2ref_C_y = 1.25

#Ergebnislisten und Nominalwerte
FR1_nom = 0.5
FR1_min = []
FR1_max =[]

FR2_nom = 0.15
FR2_min = []
FR2_max =[]

# --- Gleichungen ---

#direkte/lineare Einflüsse
#Toleranzfeld beschreibt Durchmesser, Betrachtungen hier am Radius: 1/2; jeweils halbe Toleranz +/-: 1/2; gesamt_ 1/4
#Ursache: CETOL kann keine Zylindirzität, deshalb überall Maßtoleranz genutzt

# R1 auf FR1
FR1_R1_min = - R1/4    
FR1_R1_max = R1/4
FR1_min.append(FR1_R1_min)
FR1_max.append(FR1_R1_max)
# S2 auf FR1
FR1_S2_min = - S2/4
FR1_S2_max = S2/4
FR1_min.append(FR1_S2_min)
FR1_max.append(FR1_S2_max)
# C5 auf FR2
FR2_C5_min = - C5/4
FR2_C5_max = C5/4
FR2_min.append(FR2_C5_min)
FR2_max.append(FR2_C5_max)
# N2 auf FR2
FR2_N2_min = - N2/4
FR2_N2_max = N2/4
FR2_min.append(FR2_N2_min)
FR2_max.append(FR2_N2_max)

# Einflüsse Verkippen 2D
# Die nachfolgenden Toleranzen sind Profil oder Lauftoleranzen, hier gilt jetzt wieder T_n/2 als halbe Toleranzzone

# Tilt C4 (Reines Kippen um O, C4 vershciebt in y Richtung am Mittelpunkt B2)
t = C4/2    # Translation in y Richtung gleich halbe Toleranzzonenbreite für Profil C4
# Vermutung CETOl (Screenshot fehlt, zahlen auch nicht ganz perfekt, aber leicht zu beschreiben und nah genug dran)
alpha_C4 = math.asin(t/(l_R+r_B))

# C4 auf FR1
FR1_C4_max = Pos_FR1_R_y*(1-math.cos(alpha_C4)) + Pos_FR1_R_x*math.sin(alpha_C4) # max geht nach oben in globalem y
FR1_C4_min = - FR1_C4_max
FR1_min.append(FR1_C4_min)
FR1_max.append(FR1_C4_max)
# C4 auf FR2
FR2_C4_max = Pos_FR2_RN_y*(1-math.cos(alpha_C4)) + Pos_FR2_RN_x*math.sin(alpha_C4)
FR2_C4_min = - FR2_C4_max
FR2_min.append(FR2_C4_min)
FR2_max.append(FR2_C4_max)

# Tilt S4 (rein aus S4-rotation der Positionstoleranz der Mantelfläche, Translation wegen Inkonsistenz 0 gesetzt)
# Herleitung Kinematik in Fig5 anpassen (siehe Schmierblatt)
beta_S4 = math.atan(S4/Char_L_S4) # der Drehwinkel von C in CETOLS Kinematik ist etwa 1.6*beta (in Paper irgendwie argumentativ abfangen oder Kinematik nochmal neu ansetzen)
alpha_S4 = math.acos((a_C*math.cos(beta_S4)+l_R+r_B-a_C)/(l_R+r_B))
# S4 auf FR1
FR1_S4_max = Pos_FR1_R_y*(1-math.cos(alpha_S4)) + Pos_FR1_R_x*math.sin(alpha_S4)
if S4_negative: FR1_S4_max = -FR1_S4_max
FR1_S4_min = - FR1_S4_max
FR1_min.append(FR1_S4_min)
FR1_max.append(FR1_S4_max)
# S4 auf FR2 
# hier kippt auch C mit, der Einfluss auf das Maß ist nur die relativbewegung von N zu C
# Ansatz: berechne globale Koordinaten FR2-RN (Messpunkt M) durch rotation um alpha um O und FR2-C (Bezugspunkt B) durch rotation um beta um P bzw. B2 -> Berechne Abstand zwischen Koordinaten
# Berechne über zwischengrößen Punkte M und B
# Komponente RN dreht um O um alpha_S4 und ergibt Messpunkt M
XM = Pos_FR2_RN_x * math.cos(alpha_S4) - Pos_FR2_RN_y * math.sin(alpha_S4)
YM = Pos_FR2_RN_x * math.sin(alpha_S4) + Pos_FR2_RN_y * math.cos(alpha_S4)
# Bezugspunkt B ergibt sich aus Drehung Vektor in RN um alpha + Vektor in C gedreht um beta
XB = Pos_P_R_x * math.cos(alpha_S4) - Pos_P_R_y * math.sin(alpha_S4) + Pos_FR2ref_C_x * math.cos(beta_S4) - Pos_FR2ref_C_y * math.sin(beta_S4)
YB = Pos_P_R_x * math.sin(alpha_S4) + Pos_P_R_y * math.cos(alpha_S4) + Pos_FR2ref_C_x * math.sin(beta_S4) + Pos_FR2ref_C_y * math.cos(beta_S4)
# FR2 ist Abstand zwischen beiden Raumpunkten B und M:
FR2_S4_max = math.sqrt((XB - XM)**2 + (YB - YM)**2)
if S4_negative: FR2_S4_max = -FR2_S4_max
FR2_S4_min = - FR2_S4_max
FR2_min.append(FR2_S4_min)
FR2_max.append(FR2_S4_max)

# Wiggle R4 auf FR2 (riene Rotation aus Lauftoleranz R4, Translation wegen Inkonsistenz 0 gesetzt)
gamma_R4 = math.atan(R4/Char_L_R4)
FR2_R4_max = Pos_FR2_N_y*(1-math.cos(gamma_R4)) + Pos_FR2_N_x*math.sin(gamma_R4)
FR2_R4_min = - FR2_R4_max
FR2_min.append(FR2_R4_min)
FR2_max.append(FR2_R4_max)

# --- Ausgabe ---
FR1_min_out = FR1_nom + sum(FR1_min)
FR1_max_out = FR1_nom + sum(FR1_max)
print("- FR1: -")
print("Teilbeitrag: R1 S2 C4 S4")
print(FR1_min, FR1_max)
print("Gesamtbeitrag:", sum(FR1_min), sum(FR1_max))
print("Grenzwerte:", FR1_min_out, FR1_max_out) 

FR2_min_out = FR2_nom + sum(FR2_min)
FR2_max_out = FR2_nom + sum(FR2_max)
print("- FR2: -")
print("Teilbeitrag: C5 N2 C4 S4 R4")
print(FR2_min, FR2_max)
print("Gesamtbeitrag:", sum(FR2_min), sum(FR2_max))
print("Grenzwerte:", FR2_min_out, FR2_max_out)