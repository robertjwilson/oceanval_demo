"""Build the validation report from the matchups made by matchup.py (HTML, in ./oceanval_report)."""

import oceanval

if __name__ == "__main__":
    oceanval.validate(subregions="global", concise=False)
