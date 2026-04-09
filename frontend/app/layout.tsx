import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RetrieverBench",
  description: "Compare RAG retrieval techniques side by side",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <header className="border-b border-gray-800 px-6 py-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <h1 className="text-xl font-bold tracking-tight">
              <span className="text-blue-400">Retriever</span>Bench
            </h1>
            <p className="text-sm text-gray-500">
              Compare 11 RAG retrieval techniques
            </p>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
