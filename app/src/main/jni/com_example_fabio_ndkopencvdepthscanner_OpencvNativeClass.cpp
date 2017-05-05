#include "com_example_fabio_ndkopencvdepthscanner_OpencvNativeClass.h"

JNIEXPORT void JNICALL Java_com_example_fabio_ndkopencvdepthscanner_OpencvNativeClass_crossingsDetection
(JNIEnv *, jclass, jlong addrRgba1, jlong addrRgba2){
Mat& frame1 = *(Mat*)addrRgba1;
Mat& frame2 = *(Mat*)addrRgba2;
detect(frame1, frame2);
}

void detect(Mat& frame1, Mat& frame2){
    double ratio = 1;
    Mat img2, img3;
    cvtColor(frame1, img2, CV_BGR2GRAY);
    cvtColor(frame2, img3, CV_BGR2GRAY);
    Ptr<CLAHE> clahe = createCLAHE(2); //adaptive histogram equalization
    clahe->apply(img2, img2);
    clahe->apply(img3, img3);

    String pedestrian_crossing_name = "/storage/emulated/0/cascade.xml";
    String model_name = "/storage/emulated/0/model.png";
    Ptr<ORB> detector = ORB::create(200);
    FlannBasedMatcher matcher(new flann::LshIndexParams(6, 12, 1));
    CascadeClassifier crossing_cascade;

    if( !crossing_cascade.load( pedestrian_crossing_name ) ){ printf("--(!)Error loading face cascade\n"); return;};

    vector<Rect> crossings;

    medianBlur(img2, img2, 5);
    medianBlur(img3, img3, 5);

    crossing_cascade.detectMultiScale(img2, crossings, 1.15, 3);

    vector<KeyPoint> tmpKp;
    Mat tmp, tmpDescr, out, modelDescr;
    vector<Rect> goodMatch;

    Mat model = imread(model_name, 0);
    detector->detectAndCompute(model, noArray(), tmpKp, modelDescr);

    for(int i = 0; i < crossings.size(); i++){

        tmpKp.clear();
        tmp = img2(crossings[i]);

        float ratio_sample = (float)model.size[1]/tmp.size[1];

        if(ratio_sample < 1)
            resize(tmp, tmp, Size(), ratio_sample, ratio_sample, INTER_AREA);
        else
            resize(tmp, tmp, Size(), ratio_sample, ratio_sample, INTER_CUBIC);

        detector->detectAndCompute(tmp, noArray(), tmpKp, tmpDescr);

        if(tmpKp.size() == 0)
            continue;

        vector<DMatch> matchesSym, matchesFwd, matchesBwd;
        matcher.match(modelDescr, tmpDescr, matchesFwd);
        matcher.match(tmpDescr, modelDescr, matchesBwd);

        symmetryTest(matchesFwd, matchesBwd, matchesSym);

        if(matchesSym.size() < 3)
            continue;

        goodMatch.push_back(crossings[i]);
    }

    goodMatch = collapse_double_rect(goodMatch);

    Mat temp;

    vector<Rect> findings;

    for(int i = 0; i < goodMatch.size(); i++){

        rectangle(frame2, goodMatch[i], Scalar(255, 0, 0), 4, 4, 0);

        int pad = goodMatch[i].height*0.5;

        Rect rect(0, (goodMatch[i].y-pad > 0 ? goodMatch[i].y-pad : 0),
                  img3.cols, (goodMatch[i].y-pad > 0 ? goodMatch[i].height+pad : goodMatch[i].height));

        tmp = img3(rect);

        crossing_cascade.detectMultiScale(tmp, crossings, 1.2, 3);

        for(int j = 0 ; j < crossings.size(); j++){
            Rect objRect(crossings[j].x, crossings[j].y+goodMatch[i].y-pad,
                         crossings[j].width, crossings[j].height);

            tmp = img3(objRect);

            float ratio_sample = (float)model.size[1]/tmp.size[1];

            if(ratio_sample < 1)
                resize(tmp, tmp, Size(), ratio_sample, ratio_sample, INTER_AREA);
            else
                resize(tmp, tmp, Size(), ratio_sample, ratio_sample, INTER_CUBIC);

            detector->detectAndCompute(tmp, noArray(), tmpKp, tmpDescr);

            if(tmpKp.size() == 0)
                continue;

            vector<DMatch> matchesSym, matchesFwd, matchesBwd;
            matcher.match(modelDescr, tmpDescr, matchesFwd);
            matcher.match(tmpDescr, modelDescr, matchesBwd);

            symmetryTest(matchesFwd, matchesBwd, matchesSym);

            if(matchesSym.size() < 3)
                continue;

            findings.push_back(objRect);
        }
    }

    findings = collapse_double_rect(findings);
    for(int i = 0; i < findings.size(); i++){
        rectangle(frame2, findings[i], Scalar(0, 0, 255), 4, 4, 0);
    }

    vector< pair<Rect, Rect> > corr = correspondence(img2, img3, goodMatch, findings, detector, matcher);

    //***********************DA CANI*********************************************************
    for(int i = 0; i < corr.size(); i++){
        int pxDisparity = abs((self_disparity_pixel(img2, corr[i].first) -
                           self_disparity_pixel(img3, corr[i].second))) * (2 - ratio);
        ostringstream os;
        os << pxDisparity;
        putText(frame2, os.str(), Point(corr[i].second.x, corr[i].second.y - 20),
                FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255), 3, CV_AA);
    }

}

void symmetryTest(const std::vector<cv::DMatch> &matches1,const std::vector<cv::DMatch> &matches2,std::vector<cv::DMatch>& symMatches)
{
    symMatches.clear();
    for (vector<DMatch>::const_iterator matchIterator1= matches1.begin();matchIterator1!= matches1.end(); ++matchIterator1)
    {
        for (vector<DMatch>::const_iterator matchIterator2= matches2.begin();matchIterator2!= matches2.end();++matchIterator2)
        {
            if ((*matchIterator1).queryIdx ==(*matchIterator2).trainIdx &&(*matchIterator2).queryIdx ==(*matchIterator1).trainIdx)
            {
                symMatches.push_back(DMatch((*matchIterator1).queryIdx,(*matchIterator1).trainIdx,(*matchIterator1).distance));
                break;
            }
        }
    }
}

vector<Rect> collapse_double_rect(vector<Rect> vr){
    if(vr.size() < 2)
        return vr;
    vector<Rect> res;
    int index[vr.size()];
    for(int i = 0; i < vr.size(); i++)
        index[i] = 0;
    Rect intersect, rect_union, small;
    for(int i = 0; i < vr.size(); i++){
        for(int j = (i+1); j < vr.size(); j++){
            small = vr[i].area() < vr[j].area() ? vr[i] : vr[j];
            intersect = vr[i] & vr[j];
            rect_union = vr[i] | vr[j];
            if( intersect.area() > (0.25 * small.area())){
                res.push_back(rect_union);
                index[i] = 1;
                index[j] = 1;
                res = collapse_double_rect(res);
            }
        }
    }

    for(int i = 0; i < vr.size(); i++){
        if(index[i] == 0)
            res.push_back(vr[i]);
    }
    return res;
}

vector<pair<Rect, Rect> > correspondence(Mat im1, Mat im2, vector<Rect> r_im1, vector<Rect> r_im2, Ptr<Feature2D> detector, FlannBasedMatcher matcher){

    int pad = 5;
    vector<pair<Rect, Rect> > res;
    vector<pair<pair<int, int>, double> > distanceMat;
    for(int i = 0; i < r_im1.size(); i++){
        for(int j = 0; j < r_im2.size(); j++){
            if( (rectCenter(r_im2[j]).y < (rectCenter(r_im1[i]).y - r_im1[i].height*0.5)) ||
                (rectCenter(r_im2[j]).y > (rectCenter(r_im1[i]).y + r_im1[i].height*0.5)) ){
                distanceMat.push_back(make_pair(make_pair(i, j), numeric_limits<double>::infinity()));
                continue;
            }

            Rect r1(r_im1[i].x-pad, r_im1[i].y-pad, r_im1[i].width+pad, r_im1[i].height+pad);
            Rect r2(r_im2[j].x-pad, r_im2[j].y-pad, r_im2[j].width+pad, r_im2[j].height+pad);
            vector<DMatch> matches = getFeat(detector, matcher, im1(r1).clone(), im2(r2).clone());
            distanceMat.push_back(make_pair(make_pair(i, j), meanDistance(matches)));
        }
    }

    sort(distanceMat.begin(), distanceMat.end(), myComparison);

    for( int i = 0; i < min(r_im1.size(), r_im2.size()); i++){
        res.push_back(make_pair(r_im1[distanceMat.begin()->first.first],
                                r_im2[distanceMat.begin()->first.second]));
        for(int j = 0; j < distanceMat.size(); j++){
            if(distanceMat[j].first.first == distanceMat.begin()->first.first ||
               distanceMat[j].first.second == distanceMat.begin()->first.second){
                distanceMat[j].second = numeric_limits<double>::infinity();
            }
        }
        sort(distanceMat.begin(), distanceMat.end(), myComparison);
    }

    return res;
}

vector<DMatch> getFeat(Ptr<Feature2D> alg, FlannBasedMatcher matcher, Mat model, Mat candidate){

    float ratio_sample = (double)200/model.size[1];

    if(ratio_sample < 1)
        resize(model, model, Size(), ratio_sample, ratio_sample, INTER_AREA);
    else
        resize(model, model, Size(), ratio_sample, ratio_sample, INTER_CUBIC);

    ratio_sample = (double)model.size[1]/candidate.size[1];

    if(ratio_sample < 1)
        resize(candidate, candidate, Size(), ratio_sample, ratio_sample, INTER_AREA);
    else
        resize(candidate, candidate, Size(), ratio_sample, ratio_sample, INTER_CUBIC);

    vector<KeyPoint> c_kp, m_kp;
    Mat c_descr, m_descr;

    alg->detectAndCompute(candidate, noArray(), c_kp, c_descr);
    alg->detectAndCompute(model, noArray(), m_kp, m_descr);

    vector<DMatch> matchesSym, matchesFwd, matchesBwd;
    matcher.match(model, candidate, matchesFwd);
    matcher.match(candidate, model, matchesBwd);

    symmetryTest(matchesFwd, matchesBwd, matchesSym);

    return matchesSym;
}

double meanDistance(vector<DMatch> match){
    int sum = 0;
    for(int i = 0; i < match.size(); i++)
        sum += match[i].distance;
    return (double) sum/match.size();
}

Point rectCenter(Rect r){
    return Point((r.x + r.width/2), (r.y - r.height/2));
}

bool myComparison(const pair<pair<int,int>, double> &a, const pair<pair<int,int>, double> &b)
{
    return a.second < b.second;
}

int self_disparity_pixel(Mat img, Rect r){
    int rect_center = rectCenter(r).x;
    int img_center = img.cols/2;
    return img_center - rect_center;
}