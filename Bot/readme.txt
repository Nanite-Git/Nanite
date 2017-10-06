- Com_MySQL_001.py (Stand 04.10.2017):
	Beinhaltet grob die Kommunikation zur SQL Datenbank
	Die Parameter zur Verbindung sind in der MyConnection.ini
	ausgelagert.
	
- Working_SVM_rev05.py (Stand 04.10.2017):
	Beinhaltet die komplette Funktionalität um mit 
	aktuell einer CSV-Datei Trainingsdaten einzulesen,
	daraus Features zu generieren, und mit einer
	SVM die nächste Kerze vorherzusagen.
	Aktuell ist sind die Parameter C und Gamma für die
	SVM, für Bitcoin/US-Dollar ausgelegt. Heißt:
	wenn andere History Daten verwendet werden, muss 
	zur optimalen Einstellung an den Parametern geschraubt
	werden.
	
- Beide Python Programme beinhalten momentan eine Main-Routine.
    In Zukunft sollen die Programme natürlich unter einer Routine
    laufen. Evtl. könnte man die SQL-Kommunikation und die Features
    Generierung und sonstigen Mist in Libs auslagern oder eben
    eine geeignete Klassenstruktur aufbauen. Kostet eben Zeit.
    Aktuell ist alles Quick and Dirty.

- Nanite_MetaTrader5.mq5 (Stand 04.10.2017):
	Beinhaltet das Programm im Metatrader. Dieses muss in unter
	den Dateipfad ../MQL5/Experts/ kopiert werden. Die Libs dazu
	siehe /libs/readme.txt

- /dataset/ beinhaltet aktuell noch die .csv-Dateien