import Link from "next/link";

export default function Home() {
  return (
    <div className="font-sans grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            RAGinDocs v2.0
          </h1>
          <p className="text-lg text-muted-foreground">
            Multimodal Document Processing with Advanced RAG
          </p>
          <div className="text-sm text-yellow-600 font-medium">
            🚧 Under Development - Multimodal Upgrade in Progress
          </div>
        </div>

        <div className="flex gap-4 items-center flex-col sm:flex-row">
          <Link
            className="rounded-full border border-solid border-gray-300 transition-colors flex items-center justify-center bg-gray-100 text-gray-800 gap-2 hover:bg-gray-200 font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:w-auto opacity-50 cursor-not-allowed"
            href="#"
          >
            🧪 Backend Tester (Disabled)
          </Link>
          <Link
            className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 w-full sm:w-auto md:w-[158px] opacity-50 cursor-not-allowed"
            href="#"
          >
            🚀 Main App (Coming Soon)
          </Link>
        </div>

        <div className="text-center text-sm text-muted-foreground space-y-2">
          <p>🔄 Upgrading to Multimodal RAG v2.0</p>
          <p>📊 Features: Text + Tables + Images</p>
          <p>🤖 Google Gemini 1.5 Flash Integration</p>
        </div>
      </main>
    </div>
  );
}
