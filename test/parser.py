import xml.etree.ElementTree as ET
import sys 

THRESHOLD = 0.9

if __name__ == "__main__":
    tree = ET.parse("coverage.xml")
    root = tree.getroot()
    line = root.find("./packages/package")
    if float(line.attrib["line-rate"]) >= THRESHOLD:
        sys.exit(0)
    else:
        sys.exit(1)
