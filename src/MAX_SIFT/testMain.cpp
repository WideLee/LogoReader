#include "test_max_sift.hpp"

void matchAndShow(Mat mat1, Mat mat2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, Mat des1, Mat des2, string title) {
    std::vector<DMatch> matches;
    FlannBasedMatcher matcher;

    matcher.match( des1, des2, matches );
    double max_dist = 0; double min_dist = 100000;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < des1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    printf("-- Matches: %ld\n", matches.size());

    std::vector< DMatch > good_matches;
    for( int i = 0; i < des1.rows; i++ )
    { if( matches[i].distance <= max(1.5*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
    }
    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( mat1, keypoints1, mat2, keypoints2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
    imshow( title, img_matches );
}


int main() {
    MaxSIFT MAXSIFT;
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

    string img1 = "/Users/limkuan/Study/Research/IndoorLocalization/Resource/CarLogos51/Peugeot/1316.jpg";
    string img2 = "/Users/limkuan/Desktop/1775.jpg";

    Mat mat1 = imread(img1, IMREAD_GRAYSCALE);
    Mat mat2 = imread(img2, IMREAD_GRAYSCALE);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    f2d->detect(mat1, keypoints1);
    f2d->compute(mat1, keypoints1, descriptors1);

    f2d->detect(mat2, keypoints2);
    f2d->compute(mat2, keypoints2, descriptors2);

    Mat maxsift1 = MAXSIFT.generate_max_sift_descriptor(descriptors1);
    Mat maxsift2 = MAXSIFT.generate_max_sift_descriptor(descriptors2);

    matchAndShow(mat1, mat2, keypoints1, keypoints2, maxsift1, maxsift2, "MaxSift");
    matchAndShow(mat1, mat2, keypoints1, keypoints2, descriptors1, descriptors2, "Sift");

    waitKey(0);
}


int _main()
{
	int number = 0;
	cin >> number;
    TestMaxSIFT test_maxsift(number);
    string path = "/Users/limkuan/Study/Research/IndoorLocalization/Resource/CarLogos51";
    
	vector<float> re;
    vector<vector<string> > all = test_maxsift.getAllFiles(path);

    vector<Mat> totalDescriptors = test_maxsift.getBaseDes(all);
	int numberOfRightBySift = 0, eachLogoBySift = 0;
    int numberOfRightByMax = 0, eachLogoByMax = 0;
	int sum = 0;
	ofstream fout;


	fout.open("result24v1");
	cout << all[0][0] << endl;
	for (int i = 0; i < 51; ++i) {
        eachLogoByMax = 0;
        eachLogoBySift = 0;
        // if (i == 0) continue;
        cout << i << endl;
        // fout << i << endl;
        for (int j = test_maxsift.QUERT_SIZE; j < all[i].size(); ++j) {
            cout << "\t" << j << endl;
            int pos1, pos2;
            Mat img_test = imread(all[i][j]);
            pos1 = test_maxsift.SiftCompareWithAllBasePicture(img_test, totalDescriptors);
			pos2 = test_maxsift.MaxsiftComCompareWithAllBasePicture(img_test, totalDescriptors);
			
            if (pos1 == i) {
                ++numberOfRightBySift;
                ++eachLogoBySift;
            }
            if (pos2 == i) {
                ++numberOfRightByMax;
                ++eachLogoByMax;
            }
        }
        float r1 = eachLogoBySift / (float) all[i].size();
        float r2 = eachLogoByMax / (float) all[i].size();
		sum += all[i].size();
        fout << i << " SIFT  " << r1 << "  " << numberOfRightBySift << endl;
		fout << i << " MAX-SIFT  " << r2 << "  " << numberOfRightByMax << endl;
		
		
	}
	
	fout << "SIFT Final Accuracy " << numberOfRightBySift / (float) sum << endl;
	fout << "MAX-SIFT Final Accuracy " << numberOfRightByMax / (float) sum << endl;
	fout << flush;
	fout.close();
    
 	
    waitKey(0);
}
