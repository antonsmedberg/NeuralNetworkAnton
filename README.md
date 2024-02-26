# Neural Network Project

## Projektbeskrivning

Detta projekt implementerar ett enkelt neuralt nätverk som kan användas för binär klassificering eller regression. Projektet omfattar grundläggande neurala nätverksoperationer såsom initialisering av vikter, framåtpropagering, bakåtpropagering, batchträning, och gör förutsägelser. Dessutom inkluderar projektet verktyg för att generera syntetisk data och förbereda riktig data för träning.

## Förutsättningar

För att använda detta projekt behöver du:

- Python 3.6 eller senare
- NumPy
- Pandas
- scikit-learn

## Installation och Konfiguration

Inga ytterligare installationer krävs utöver de ovan nämnda biblioteken. Kontrollera att din Python-miljö är korrekt konfigurerad och att alla nödvändiga bibliotek är installerade.

## Hur man kör koden

För att köra projektet, öppna en terminal, navigera till projektets rotkatalog, och kör följande kommando:

```bash
python main.py
```

## Användning med Syntetisk Data och Riktig Data

Detta projekt stödjer användning med både syntetiskt genererad data och riktig data. För att växla mellan dessa, ändra `use_synthetic_data`-flaggan i `config.json`:

- För syntetisk data, sätt `"use_synthetic_data": true`
- För riktig data, sätt `"use_synthetic_data": false` och specificera sökvägen till din datafil i `"filepath"`

## Förklaring av Kodens Arkitektur

- `NeuralNetwork`: Kärnklassen som implementerar det neurala nätverket.
- `generate_binary_classification_data`: Funktion för att generera syntetisk data.
- `load_real_data`: Funktion för att ladda och förbereda riktig data från en CSV-fil.
- `train_and_evaluate`: Funktion för att träna nätverket och utvärdera dess prestanda.
- `plot_losses` och `plot_decision_boundary`: Funktioner för visuell feedback på nätverkets lärande och dess beslutsgränser.

## Bidra till Projektet

Alla bidrag till projektet uppmuntras och uppskattas. För att bidra, följ de vanliga procedurerna för att skapa en pull-förfrågan eller öppna en issue för diskussion kring större förändringar eller förbättringar.
