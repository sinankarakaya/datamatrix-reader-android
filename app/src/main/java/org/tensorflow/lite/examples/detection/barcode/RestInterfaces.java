package org.tensorflow.lite.examples.detection.barcode;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.POST;
import retrofit2.http.Path;

public interface RestInterfaces {

    @FormUrlEncoded
    @POST("eczanet/qrcodeController")
    Call<BarcodeResponse> sendBarcodeList(@Field("qrcodes") String qrcodes);


}
