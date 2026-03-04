import type { Metadata } from "next";
import "@/styles/terminal.css";

export const metadata: Metadata = {
  title: "Polymarket Alpha Terminal",
  description: "Terminal de trading quantitativo para Polymarket",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body className="bg-terminal-bg text-terminal-text font-mono antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}
