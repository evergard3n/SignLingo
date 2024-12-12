"use client";

import { toast } from "sonner";
import Image from "next/image";
import Confetti from "react-confetti";
import { useRouter } from "next/navigation";
import { useCallback, useState, useTransition } from "react";
import { useAudio, useWindowSize, useMount } from "react-use";

import { reduceHearts } from "@/actions/user-progress";
import { useHeartsModal } from "@/store/use-hearts-modal";
import { challengeOptions, challenges } from "@/db/schema";
import { usePracticeModal } from "@/store/use-practice-modal";
import { upsertChallengeProgress } from "@/actions/challenge-progress";

import { Header } from "./header";
import { Footer } from "./footer";
import { Challenge } from "./challenge";
import { ResultCard } from "./result-card";
import { LearningBubble, QuestionBubble } from "./question-bubble";
import { Button } from "@/components/ui/button";
import { log } from "console";

type Props = {
  initialPercentage: number;
  initialHearts: number;
  initialLessonId: number;
  initialLessonChallenges: (typeof challenges.$inferSelect & {
    completed: boolean;
    challengeOptions: (typeof challengeOptions.$inferSelect)[];
  })[];
};

export const Quiz = ({
  initialPercentage,
  initialHearts,
  initialLessonId,
  initialLessonChallenges,
}: Props) => {
  const { open: openHeartsModal } = useHeartsModal();
  const { open: openPracticeModal } = usePracticeModal();

  useMount(() => {
    if (initialPercentage === 100) {
      openPracticeModal();
    }
  });

  const { width, height } = useWindowSize();

  const router = useRouter();

  const [finishAudio] = useAudio({ src: "/finish.mp3", autoPlay: true });
  const [correctAudio, _c, correctControls] = useAudio({ src: "/correct.wav" });
  const [incorrectAudio, _i, incorrectControls] = useAudio({
    src: "/incorrect.wav",
  });
  const [pending, startTransition] = useTransition();

  const [lessonId] = useState(initialLessonId);
  const [hearts, setHearts] = useState(initialHearts);
  const [percentage, setPercentage] = useState(() => {
    return initialPercentage === 100 ? 0 : initialPercentage;
  });
  const [challenges] = useState(initialLessonChallenges);
  const [activeIndex, setActiveIndex] = useState(() => {
    const uncompletedIndex = challenges.findIndex(
      (challenge) => !challenge.completed
    );
    return uncompletedIndex === -1 ? 0 : uncompletedIndex;
  });

  const [selectedOption, setSelectedOption] = useState<number>();
  const [status, setStatus] = useState<"correct" | "wrong" | "none">("none");
  const [hintOpen, setHintOpen] = useState(false);

  const challenge = challenges[activeIndex];
  const options = challenge?.challengeOptions ?? [];

  const onNext = () => {
    setActiveIndex((current) => current + 1);
  };

  const onSelect = (id: number) => {
    if (status !== "none") return;

    setSelectedOption(id);
    console.log("selected", selectedOption);
  };
  const onContinueWithId = (selectedOption:number)=>{
    console.log("continue");
    if (!selectedOption) {
      
      console.log("no selected option");
      return;}

    if (status === "wrong") {
      setStatus("none");
      console.log("wrong");
      
      setSelectedOption(undefined);
      return;
    }

    if (status === "correct") {
      onNext();
      setStatus("none");
      console.log("correct");
      
      setSelectedOption(undefined);
      return;
    }

    const correctOption = options.find((option) => option.correct);
    console.log("correctOption", correctOption);
    
    if (!correctOption) {
      return;
    }

    if (correctOption.id === selectedOption) {
      startTransition(() => {
        upsertChallengeProgress(challenge.id)
          .then((response) => {
            if (response?.error === "hearts") {
              openHeartsModal();
              return;
            }

            correctControls.play();
            setStatus("correct");
            setPercentage((prev) => prev + 100 / challenges.length);

            // This is a practice
            if (initialPercentage === 100) {
              setHearts((prev) => Math.min(prev + 1, 5));
            }
          })
          .catch(() => toast.error("Something went wrong. Please try again."));
      });
    } else {
      startTransition(() => {
        reduceHearts(challenge.id)
          .then((response) => {
            if (response?.error === "hearts") {
              openHeartsModal();
              return;
            }

            incorrectControls.play();
            setStatus("wrong");

            if (!response?.error) {
              setHearts((prev) => Math.max(prev - 1, 0));
            }
          })
          .catch(() => toast.error("Something went wrong. Please try again."));
      });
    }
  };
  const onContinue = ()=>{
    console.log("continue");
    if (!selectedOption) {
      
      console.log("no selected option");
      
      return;}

    if (status === "wrong") {
      setStatus("none");
      console.log("wrong");
      
      setSelectedOption(undefined);
      return;
    }

    if (status === "correct") {
      onNext();
      setStatus("none");
      console.log("correct");
      
      setSelectedOption(undefined);
      return;
    }

    const correctOption = options.find((option) => option.correct);
    console.log("correctOption", correctOption);
    
    if (!correctOption) {
      return;
    }

    if (correctOption.id === selectedOption) {
      startTransition(() => {
        upsertChallengeProgress(challenge.id)
          .then((response) => {
            if (response?.error === "hearts") {
              openHeartsModal();
              return;
            }

            correctControls.play();
            setStatus("correct");
            setPercentage((prev) => prev + 100 / challenges.length);

            // This is a practice
            if (initialPercentage === 100) {
              setHearts((prev) => Math.min(prev + 1, 5));
            }
          })
          .catch(() => toast.error("Something went wrong. Please try again."));
      });
    } else {
      startTransition(() => {
        reduceHearts(challenge.id)
          .then((response) => {
            if (response?.error === "hearts") {
              openHeartsModal();
              return;
            }

            incorrectControls.play();
            setStatus("wrong");

            if (!response?.error) {
              setHearts((prev) => Math.max(prev - 1, 0));
            }
          })
          .catch(() => toast.error("Something went wrong. Please try again."));
      });
    }
  };

  if (!challenge) {
    return (
      <>
        {finishAudio}
        <Confetti
          width={width}
          height={height}
          recycle={false}
          numberOfPieces={500}
          tweenDuration={10000}
        />
        <div className="flex flex-col gap-y-4 lg:gap-y-8 max-w-lg mx-auto text-center items-center justify-center h-full">
          <Image
            src="/finish.svg"
            alt="Finish"
            className="hidden lg:block"
            height={100}
            width={100}
          />
          <Image
            src="/finish.svg"
            alt="Finish"
            className="block lg:hidden"
            height={50}
            width={50}
          />
          <h1 className="text-xl lg:text-3xl font-bold text-neutral-700">
            Great job! <br /> You&apos;ve completed the lesson.
          </h1>
          <div className="flex items-center gap-x-4 w-full">
            <ResultCard variant="points" value={challenges.length * 10} />
            <ResultCard variant="hearts" value={hearts} />
          </div>
        </div>
        <Footer
          lessonId={lessonId}
          status="completed"
          onCheck={() => router.push("/learn")}
        />
      </>
    );
  }

  let title;
  switch (challenge.type) {
    case "ASSIST":
      title = "Chọn từ được biểu diễn trong video:";
      break;
    case "LEARN":
      if (challenge.lessonId != 6) {
        title = "Từ mới: " + challenge.question;
      } else {
        title = "Kiểm tra kí hiệu: " + challenge.question;
      }

      break;
    default:
      title = challenge.question;
  }
  let className: string;
  if (challenge.lessonId == 6) {
    className =
      "flex flex-row gap-2 h-[300px] items-center justify-center";
  } else {
    className = "";
  }
  return (
    <>
      {incorrectAudio}
      {correctAudio}
      <Header hearts={hearts} percentage={percentage} />
      <div className="flex-1 relative">
        <div className="h-full flex items-center justify-center">
          <div className="lg:min-h-[350px] lg:w-[600px] w-full px-6 lg:px-0 flex flex-col gap-y-12">
            <div className="flex flex-row items-center justify-between">
              <h1 className="text-lg lg:text-3xl text-center lg:text-start font-bold text-neutral-700">
                {title}
              </h1>
              <Button
                className="p-2 border-2 rounded-xl"
                variant="ghost"
                onClick={() => {
                  setHintOpen(!hintOpen);
                }}
              >
                Hint
              </Button>
            </div>
            {hintOpen && (
              <div className={`absolute border-2 rounded-xl p-2 w-1/4 h-fit bg-white top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 transition duration-300 ease-linear ${
                hintOpen ? "opacity-100" : "opacity-0"
              }`}>
                <button
                className="p-2 border-2 rounded-xl"
                onClick={() => {
                  setHintOpen(!hintOpen);
                }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" className="lucide lucide-x text-slate-500 hover:opacity-75 transition cursor-pointer"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>
              </button>
                <LearningBubble vidsrc={options.find((option) => option.correct)?.imageSrc} />
              </div>
            )}
            <div className={className}>
              {challenge.type === "ASSIST" && (
                <LearningBubble
                  vidsrc={options.find((option) => option.correct)?.imageSrc}
                />
              )}
              {challenge.type === "LEARN" && challenge.lessonId != 6 && (
                <LearningBubble vidsrc={options.at(0)?.imageSrc} />
              )}

              <Challenge
                options={options}
                onSelect={onSelect}
                status={status}
                selectedOption={selectedOption}
                disabled={pending}
                type={challenge.type}
                lessonId={challenge.lessonId}
                onCheck= {onContinueWithId}
              />
            </div>
          </div>
        </div>
      </div>
      <Footer
        disabled={pending || !selectedOption}
        status={status}
        onCheck={onContinue}
      />
    </>
  );
};
