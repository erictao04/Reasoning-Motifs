import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { LandingPage } from "./pages/LandingPage";
import { QuestionDetailPage } from "./pages/QuestionDetailPage";
import { QuestionExplorerPage } from "./pages/QuestionExplorerPage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/questions" element={<QuestionExplorerPage />} />
        <Route path="/questions/:questionId" element={<QuestionDetailPage />} />
      </Routes>
    </AppShell>
  );
}
