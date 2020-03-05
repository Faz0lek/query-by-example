ARCHIVE=xkoste12.tar.gz
FILES=$(wildcard src/*.py) $(wildcard sentences/*.wav) $(wildcard queries/*.wav) $(wildcard hits/*.wav)

${ARCHIVE}: ${FILES}
	tar czf ${ARCHIVE} ${FILES}
