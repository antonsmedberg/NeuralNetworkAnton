# Projekt Neurala Nätverk

## Projektöversikt

Detta projekt implementerar ett grundläggande ramverk för neurala nätverk som kan användas för binär klassificering eller regression. Projektet inkluderar funktioner för att initialisera vikter, framåtpropagering, bakåtpropagering, träning med batcher och att göra förutsägelser. Dessutom tillhandahålls verktyg för att generera syntetisk data och förbereda riktig data för träning.
## Förutsättningar

För att använda detta projekt behöver du:

- Python 3.6 eller senare
- NumPy
- Pandas
- scikit-learn
- Matplotlib (för visualisering av data)

## Installation och Konfiguration

Inga ytterligare installationer krävs utöver de nämnda biblioteken. Kontrollera att din Python-miljö är korrekt konfigurerad och att alla nödvändiga bibliotek är installerade.

## Hur man kör koden

För att köra projektet, öppna en terminal, navigera till projektets rotkatalog och kör följande kommando:

```bash
python main.py
```

## Användning med Syntetisk och Riktig Data

Projektet stödjer arbete med både syntetisk och riktig data. För att växla mellan dessa, justera use_synthetic_data-flaggan i config.json:

- För syntetisk data, sätt `"use_synthetic_data": true`
- För riktig data, sätt `"use_synthetic_data": false` och specificera sökvägen till din datafil i `"filepath"`

  Se till att target_column i funktionen load_real_data matchar målfunktionen i din dataset.


## Förklaring av Kodens Arkitektur

- `NeuralNetwork`: Kärnklassen som implementerar det neurala nätverket.
- `generate_data.py`: Innehåller funktioner för att generera syntetisk data och ladda riktig data. Detta underlättar enkel växling mellan datatyper.
- `load_real_data`: Funktion för att ladda och förbereda riktig data från en CSV-fil.
- `train_and_evaluate`: Funktion för att träna nätverket och utvärdera dess prestanda.
- `plot_losses` och `plot_decision_boundary`: Funktioner för visuell feedback på nätverkets lärande och dess beslutsgränser.

## Anpassning och Parametrar

Du kan anpassa neurala nätverkets arkitektur och träningsprocess genom att modifiera parametrar i
`config.json`,
`layer_sizes`,
`learning_rate`,
`epochs` med mera.

## Visualisera Data

Projektet inkluderar en funktion plot_generated_data för att visualisera syntetisk data. Detta kan vara användbart för att förstå datadistributionen och separationen mellan klasserna.


## Bidra till Projektet

Alla bidrag till projektet uppmuntras och uppskattas. För att bidra, följ de vanliga procedurerna för att skapa en pull-förfrågan eller öppna en issue för diskussion kring större förändringar eller förbättringar.

## Licens

Detta projekt är licensierat under MIT-licensen. Se [LICENSE](LICENSE) filen för mer information.
