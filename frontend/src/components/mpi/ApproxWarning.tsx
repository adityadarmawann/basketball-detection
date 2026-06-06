export default function ApproxWarning() {
  return (
    <div className="bg-warning/10 border-l-4 border-warning rounded-lg p-4 mb-6">
      <div className="flex gap-3">
        <span className="text-2xl">⚠️</span>
        <div>
          <h3 className="font-bold text-warning mb-1">ESTIMASI — Data berbasis Computer Vision</h3>
          <p className="text-sm text-text-secondary">
            Data dalam halaman ini diestimasi menggunakan Computer Vision, bukan sensor IMU/GPS. 
            Akurasi ±15-25%. Gunakan hanya sebagai referensi analisis performa relatif.
          </p>
        </div>
      </div>
    </div>
  )
}
