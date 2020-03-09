package org.tensorflow.lite.examples.detection.barcode;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import org.tensorflow.lite.examples.detection.R;
import java.util.ArrayList;
import java.util.List;

public class BarcodeAdapter extends RecyclerView.Adapter<BarcodeAdapter.BarcodeHolder> {

    Context context;
    List<String> list;

    public class BarcodeHolder extends RecyclerView.ViewHolder {
        public TextView title;
        public BarcodeHolder(View view) {
            super(view);
            title = (TextView) view.findViewById(R.id.barcode);
        }
    }

    public BarcodeAdapter(Context context, List<String> list) {
        this.context = context;
        this.list = list;
    }

    @NonNull
    @Override
    public BarcodeHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.tfe_list_barcode_row, parent, false);
        return new BarcodeHolder(itemView);
    }

    @Override
    public void onBindViewHolder(@NonNull BarcodeHolder holder, int position) {
        holder.title.setText(list.get(position));
    }

    @Override
    public int getItemCount() {
        return list.size();
    }
}
