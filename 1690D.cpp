# include <bits/stdc++.h>


using namespace std;

int solve(int n, int k,string s){

	int ans = 0;
	int counter = 0;
	int minn = 1e9;

	vector<int> arr(n + 1);

	for (int i=1; i <= n; i++){
		if (s[i - 1] == 'W'){
			counter++;
		}
		arr[i] = counter;
	}

	for (int i=1; i + k - 1 <= n; i++){
		minn = min(minn, arr[i+k-1] - arr[i-1]);
	}
	return minn;


}

int main(){
	int t;
	cin >> t;

	while (t--){
		int n, k;
		string s;
		cin >> n >> k >> s;

		cout << solve(n, k, s) << endl;
	}

}