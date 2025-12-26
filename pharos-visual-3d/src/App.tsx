import { HeroUIProvider } from "@heroui/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Route } from "react-router";
import { Routes } from "react-router";

import { HelpPage } from "@/pages/help";
import { HomePage } from "@/pages/home";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <HeroUIProvider>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/help" element={<HelpPage />} />
        </Routes>
      </HeroUIProvider>
    </QueryClientProvider>
  );
}

export default App;
