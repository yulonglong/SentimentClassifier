#include <bits/stdc++.h>
#include <dirent.h>
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()
using namespace std;

void printHelp() {
	cout << "\n" << 
			"================== combineText4word2vec HELP =====================\n" <<
			"This script is made to combine all the text files to be used with word2vec for word embedding training.\n" << 
			"1. First argument is the path to the folder containing the text files\n" << 
			"   => e.g., \"data/aclImdb/unsup/\"\n" <<
			"2. Second argument is the path to text file to be created containing all of the combined ED notes text.\n" << 
			"   => e.g., \"data/combinedEDNotes.txt\"\n" <<
			"\n" <<
			"Example of a full command:\n" <<
			" \"./combineText4word2vec  data/aclImdb/train/unsup/ word2vec/vectors/temp_combinedReviews.txt\"\n" <<
			"=================================================================\n"
			<< endl;
}

int getdir (string dir, vector<string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(dir.c_str())) == NULL) {
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) {
		files.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

void readAndWrite(string inputFilename, ofstream& outfile) {
	ifstream infile;
	infile.open(inputFilename.c_str(),ios::in);
	if (infile.fail()) {
		cout << "Failed to open file : " << inputFilename << endl;
	}
	else {
		string line;
		// read and write every line to the new file
		while (getline(infile,line)) {
			outfile << line << endl;
		}
	}
	infile.close();
	return;
}

int main (int argc, char* args[]) {
	if (argc != 3) {
		cout << "Please enter correct number of arguments!" << endl;
		printHelp();
		return 0;
	}

	string folderPath = args[1];
	string combinedTextFilename = args[2];

	ofstream outfile;
	outfile.open(combinedTextFilename.c_str(),ios::out);
	if (outfile.fail()) {
		cout << "Failed to create : " << combinedTextFilename << endl;
		return 0;
	}

	vector<string> files;

	files.clear();
	string path = folderPath;
	int counter = 0;
	getdir(path,files);
	for (unsigned int i = 0;i < files.size();i++) {
		if (files[i]=="." || files[i]=="..") continue;

		string currentFilePath = path+"/"+files[i];
		readAndWrite(currentFilePath, outfile);
		counter++;
	}
	cout << path << " - Total files combined : " << counter << endl;
	
	outfile.close();
	return 0;
}