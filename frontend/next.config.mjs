/** Static export: `npm run build` writes ./out, which the FastAPI server can serve directly. */
const nextConfig = { output: "export", reactStrictMode: true, images: { unoptimized: true } };
export default nextConfig;
